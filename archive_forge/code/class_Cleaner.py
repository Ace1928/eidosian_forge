import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
class Cleaner:
    """
    Instances cleans the document of each of the possible offending
    elements.  The cleaning is controlled by attributes; you can
    override attributes in a subclass, or set them in the constructor.

    ``scripts``:
        Removes any ``<script>`` tags.

    ``javascript``:
        Removes any Javascript, like an ``onclick`` attribute. Also removes stylesheets
        as they could contain Javascript.

    ``comments``:
        Removes any comments.

    ``style``:
        Removes any style tags.

    ``inline_style``
        Removes any style attributes.  Defaults to the value of the ``style`` option.

    ``links``:
        Removes any ``<link>`` tags

    ``meta``:
        Removes any ``<meta>`` tags

    ``page_structure``:
        Structural parts of a page: ``<head>``, ``<html>``, ``<title>``.

    ``processing_instructions``:
        Removes any processing instructions.

    ``embedded``:
        Removes any embedded objects (flash, iframes)

    ``frames``:
        Removes any frame-related tags

    ``forms``:
        Removes any form tags

    ``annoying_tags``:
        Tags that aren't *wrong*, but are annoying.  ``<blink>`` and ``<marquee>``

    ``remove_tags``:
        A list of tags to remove.  Only the tags will be removed,
        their content will get pulled up into the parent tag.

    ``kill_tags``:
        A list of tags to kill.  Killing also removes the tag's content,
        i.e. the whole subtree, not just the tag itself.

    ``allow_tags``:
        A list of tags to include (default include all).

    ``remove_unknown_tags``:
        Remove any tags that aren't standard parts of HTML.

    ``safe_attrs_only``:
        If true, only include 'safe' attributes (specifically the list
        from the feedparser HTML sanitisation web site).

    ``safe_attrs``:
        A set of attribute names to override the default list of attributes
        considered 'safe' (when safe_attrs_only=True).

    ``add_nofollow``:
        If true, then any <a> tags will have ``rel="nofollow"`` added to them.

    ``host_whitelist``:
        A list or set of hosts that you can use for embedded content
        (for content like ``<object>``, ``<link rel="stylesheet">``, etc).
        You can also implement/override the method
        ``allow_embedded_url(el, url)`` or ``allow_element(el)`` to
        implement more complex rules for what can be embedded.
        Anything that passes this test will be shown, regardless of
        the value of (for instance) ``embedded``.

        Note that this parameter might not work as intended if you do not
        make the links absolute before doing the cleaning.

        Note that you may also need to set ``whitelist_tags``.

    ``whitelist_tags``:
        A set of tags that can be included with ``host_whitelist``.
        The default is ``iframe`` and ``embed``; you may wish to
        include other tags like ``script``, or you may want to
        implement ``allow_embedded_url`` for more control.  Set to None to
        include all tags.

    This modifies the document *in place*.
    """
    scripts = True
    javascript = True
    comments = True
    style = False
    inline_style = None
    links = True
    meta = True
    page_structure = True
    processing_instructions = True
    embedded = True
    frames = True
    forms = True
    annoying_tags = True
    remove_tags = ()
    allow_tags = ()
    kill_tags = ()
    remove_unknown_tags = True
    safe_attrs_only = True
    safe_attrs = defs.safe_attrs
    add_nofollow = False
    host_whitelist = ()
    whitelist_tags = {'iframe', 'embed'}

    def __init__(self, **kw):
        not_an_attribute = object()
        for name, value in kw.items():
            default = getattr(self, name, not_an_attribute)
            if default is None or default is True or default is False:
                pass
            elif isinstance(default, (frozenset, set, tuple, list)):
                if isinstance(value, str):
                    raise TypeError(f'Expected a collection, got str: {name}={value!r}')
            else:
                raise TypeError(f'Unknown parameter: {name}={value!r}')
            setattr(self, name, value)
        if self.inline_style is None and 'inline_style' not in kw:
            self.inline_style = self.style
        if kw.get('allow_tags'):
            if kw.get('remove_unknown_tags'):
                raise ValueError('It does not make sense to pass in both allow_tags and remove_unknown_tags')
            self.remove_unknown_tags = False
        self.host_whitelist = frozenset(self.host_whitelist) if self.host_whitelist else ()
    _tag_link_attrs = dict(script='src', link='href', applet=['code', 'object'], iframe='src', embed='src', layer='src', a='href')

    def __call__(self, doc):
        """
        Cleans the document.
        """
        try:
            getroot = doc.getroot
        except AttributeError:
            pass
        else:
            doc = getroot()
        xhtml_to_html(doc)
        for el in doc.iter('image'):
            el.tag = 'img'
        if not self.comments:
            self.kill_conditional_comments(doc)
        kill_tags = set(self.kill_tags or ())
        remove_tags = set(self.remove_tags or ())
        allow_tags = set(self.allow_tags or ())
        if self.scripts:
            kill_tags.add('script')
        if self.safe_attrs_only:
            safe_attrs = set(self.safe_attrs)
            for el in doc.iter(etree.Element):
                attrib = el.attrib
                for aname in attrib.keys():
                    if aname not in safe_attrs:
                        del attrib[aname]
        if self.javascript:
            if not (self.safe_attrs_only and self.safe_attrs == defs.safe_attrs):
                for el in doc.iter(etree.Element):
                    attrib = el.attrib
                    for aname in attrib.keys():
                        if aname.startswith('on'):
                            del attrib[aname]
            doc.rewrite_links(self._remove_javascript_link, resolve_base_href=False)
            if not self.inline_style:
                for el in _find_styled_elements(doc):
                    old = el.get('style')
                    new = _replace_css_javascript('', old)
                    new = _replace_css_import('', new)
                    if self._has_sneaky_javascript(new):
                        del el.attrib['style']
                    elif new != old:
                        el.set('style', new)
            if not self.style:
                for el in list(doc.iter('style')):
                    if el.get('type', '').lower().strip() == 'text/javascript':
                        el.drop_tree()
                        continue
                    old = el.text or ''
                    new = _replace_css_javascript('', old)
                    new = _replace_css_import('', new)
                    if self._has_sneaky_javascript(new):
                        el.text = '/* deleted */'
                    elif new != old:
                        el.text = new
        if self.comments:
            kill_tags.add(etree.Comment)
        if self.processing_instructions:
            kill_tags.add(etree.ProcessingInstruction)
        if self.style:
            kill_tags.add('style')
        if self.inline_style:
            etree.strip_attributes(doc, 'style')
        if self.links:
            kill_tags.add('link')
        elif self.style or self.javascript:
            for el in list(doc.iter('link')):
                if 'stylesheet' in el.get('rel', '').lower():
                    if not self.allow_element(el):
                        el.drop_tree()
        if self.meta:
            kill_tags.add('meta')
        if self.page_structure:
            remove_tags.update(('head', 'html', 'title'))
        if self.embedded:
            for el in list(doc.iter('param')):
                parent = el.getparent()
                while parent is not None and parent.tag not in ('applet', 'object'):
                    parent = parent.getparent()
                if parent is None:
                    el.drop_tree()
            kill_tags.update(('applet',))
            remove_tags.update(('iframe', 'embed', 'layer', 'object', 'param'))
        if self.frames:
            kill_tags.update(defs.frame_tags)
        if self.forms:
            remove_tags.add('form')
            kill_tags.update(('button', 'input', 'select', 'textarea'))
        if self.annoying_tags:
            remove_tags.update(('blink', 'marquee'))
        _remove = []
        _kill = []
        for el in doc.iter():
            if el.tag in kill_tags:
                if self.allow_element(el):
                    continue
                _kill.append(el)
            elif el.tag in remove_tags:
                if self.allow_element(el):
                    continue
                _remove.append(el)
        if _remove and _remove[0] == doc:
            el = _remove.pop(0)
            el.tag = 'div'
            el.attrib.clear()
        elif _kill and _kill[0] == doc:
            el = _kill.pop(0)
            if el.tag != 'html':
                el.tag = 'div'
            el.clear()
        _kill.reverse()
        for el in _kill:
            el.drop_tree()
        for el in _remove:
            el.drop_tag()
        if self.remove_unknown_tags:
            if allow_tags:
                raise ValueError('It does not make sense to pass in both allow_tags and remove_unknown_tags')
            allow_tags = set(defs.tags)
        if allow_tags:
            if not self.comments:
                allow_tags.add(etree.Comment)
            if not self.processing_instructions:
                allow_tags.add(etree.ProcessingInstruction)
            bad = []
            for el in doc.iter():
                if el.tag not in allow_tags:
                    bad.append(el)
            if bad:
                if bad[0] is doc:
                    el = bad.pop(0)
                    el.tag = 'div'
                    el.attrib.clear()
                for el in bad:
                    el.drop_tag()
        if self.add_nofollow:
            for el in _find_external_links(doc):
                if not self.allow_follow(el):
                    rel = el.get('rel')
                    if rel:
                        if 'nofollow' in rel and ' nofollow ' in ' %s ' % rel:
                            continue
                        rel = '%s nofollow' % rel
                    else:
                        rel = 'nofollow'
                    el.set('rel', rel)

    def allow_follow(self, anchor):
        """
        Override to suppress rel="nofollow" on some anchors.
        """
        return False

    def allow_element(self, el):
        """
        Decide whether an element is configured to be accepted or rejected.

        :param el: an element.
        :return: true to accept the element or false to reject/discard it.
        """
        if el.tag not in self._tag_link_attrs:
            return False
        attr = self._tag_link_attrs[el.tag]
        if isinstance(attr, (list, tuple)):
            for one_attr in attr:
                url = el.get(one_attr)
                if not url:
                    return False
                if not self.allow_embedded_url(el, url):
                    return False
            return True
        else:
            url = el.get(attr)
            if not url:
                return False
            return self.allow_embedded_url(el, url)

    def allow_embedded_url(self, el, url):
        """
        Decide whether a URL that was found in an element's attributes or text
        if configured to be accepted or rejected.

        :param el: an element.
        :param url: a URL found on the element.
        :return: true to accept the URL and false to reject it.
        """
        if self.whitelist_tags is not None and el.tag not in self.whitelist_tags:
            return False
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https'):
            return False
        if parts.hostname in self.host_whitelist:
            return True
        return False

    def kill_conditional_comments(self, doc):
        """
        IE conditional comments basically embed HTML that the parser
        doesn't normally see.  We can't allow anything like that, so
        we'll kill any comments that could be conditional.
        """
        has_conditional_comment = _conditional_comment_re.search
        self._kill_elements(doc, lambda el: has_conditional_comment(el.text), etree.Comment)

    def _kill_elements(self, doc, condition, iterate=None):
        bad = []
        for el in doc.iter(iterate):
            if condition(el):
                bad.append(el)
        for el in bad:
            el.drop_tree()

    def _remove_javascript_link(self, link):
        new = _substitute_whitespace('', unquote_plus(link))
        if _has_javascript_scheme(new):
            return ''
        return link
    _substitute_comments = re.compile('/\\*.*?\\*/', re.S).sub

    def _has_sneaky_javascript(self, style):
        """
        Depending on the browser, stuff like ``e x p r e s s i o n(...)``
        can get interpreted, or ``expre/* stuff */ssion(...)``.  This
        checks for attempt to do stuff like this.

        Typically the response will be to kill the entire style; if you
        have just a bit of Javascript in the style another rule will catch
        that and remove only the Javascript from the style; this catches
        more sneaky attempts.
        """
        style = self._substitute_comments('', style)
        style = style.replace('\\', '')
        style = _substitute_whitespace('', style)
        style = style.lower()
        if _has_javascript_scheme(style):
            return True
        if 'expression(' in style:
            return True
        if '@import' in style:
            return True
        if '</noscript' in style:
            return True
        if _looks_like_tag_content(style):
            return True
        return False

    def clean_html(self, html):
        result_type = type(html)
        if isinstance(html, (str, bytes)):
            doc = fromstring(html)
        else:
            doc = copy.deepcopy(html)
        self(doc)
        return _transform_result(result_type, doc)