import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class HtmlMixin:

    def set(self, key, value=None):
        """set(self, key, value=None)

        Sets an element attribute.  If no value is provided, or if the value is None,
        creates a 'boolean' attribute without value, e.g. "<form novalidate></form>"
        for ``form.set('novalidate')``.
        """
        super().set(key, value)

    @property
    def classes(self):
        """
        A set-like wrapper around the 'class' attribute.
        """
        return Classes(self.attrib)

    @classes.setter
    def classes(self, classes):
        assert isinstance(classes, Classes)
        value = classes._get_class_value()
        if value:
            self.set('class', value)
        elif self.get('class') is not None:
            del self.attrib['class']

    @property
    def base_url(self):
        """
        Returns the base URL, given when the page was parsed.

        Use with ``urlparse.urljoin(el.base_url, href)`` to get
        absolute URLs.
        """
        return self.getroottree().docinfo.URL

    @property
    def forms(self):
        """
        Return a list of all the forms
        """
        return _forms_xpath(self)

    @property
    def body(self):
        """
        Return the <body> element.  Can be called from a child element
        to get the document's head.
        """
        return self.xpath('//body|//x:body', namespaces={'x': XHTML_NAMESPACE})[0]

    @property
    def head(self):
        """
        Returns the <head> element.  Can be called from a child
        element to get the document's head.
        """
        return self.xpath('//head|//x:head', namespaces={'x': XHTML_NAMESPACE})[0]

    @property
    def label(self):
        """
        Get or set any <label> element associated with this element.
        """
        id = self.get('id')
        if not id:
            return None
        result = _label_xpath(self, id=id)
        if not result:
            return None
        else:
            return result[0]

    @label.setter
    def label(self, label):
        id = self.get('id')
        if not id:
            raise TypeError('You cannot set a label for an element (%r) that has no id' % self)
        if _nons(label.tag) != 'label':
            raise TypeError('You can only assign label to a label element (not %r)' % label)
        label.set('for', id)

    @label.deleter
    def label(self):
        label = self.label
        if label is not None:
            del label.attrib['for']

    def drop_tree(self):
        """
        Removes this element from the tree, including its children and
        text.  The tail text is joined to the previous element or
        parent.
        """
        parent = self.getparent()
        assert parent is not None
        if self.tail:
            previous = self.getprevious()
            if previous is None:
                parent.text = (parent.text or '') + self.tail
            else:
                previous.tail = (previous.tail or '') + self.tail
        parent.remove(self)

    def drop_tag(self):
        """
        Remove the tag, but not its children or text.  The children and text
        are merged into the parent.

        Example::

            >>> h = fragment_fromstring('<div>Hello <b>World!</b></div>')
            >>> h.find('.//b').drop_tag()
            >>> print(tostring(h, encoding='unicode'))
            <div>Hello World!</div>
        """
        parent = self.getparent()
        assert parent is not None
        previous = self.getprevious()
        if self.text and isinstance(self.tag, str):
            if previous is None:
                parent.text = (parent.text or '') + self.text
            else:
                previous.tail = (previous.tail or '') + self.text
        if self.tail:
            if len(self):
                last = self[-1]
                last.tail = (last.tail or '') + self.tail
            elif previous is None:
                parent.text = (parent.text or '') + self.tail
            else:
                previous.tail = (previous.tail or '') + self.tail
        index = parent.index(self)
        parent[index:index + 1] = self[:]

    def find_rel_links(self, rel):
        """
        Find any links like ``<a rel="{rel}">...</a>``; returns a list of elements.
        """
        rel = rel.lower()
        return [el for el in _rel_links_xpath(self) if el.get('rel').lower() == rel]

    def find_class(self, class_name):
        """
        Find any elements with the given class name.
        """
        return _class_xpath(self, class_name=class_name)

    def get_element_by_id(self, id, *default):
        """
        Get the first element in a document with the given id.  If none is
        found, return the default argument if provided or raise KeyError
        otherwise.

        Note that there can be more than one element with the same id,
        and this isn't uncommon in HTML documents found in the wild.
        Browsers return only the first match, and this function does
        the same.
        """
        try:
            return _id_xpath(self, id=id)[0]
        except IndexError:
            if default:
                return default[0]
            else:
                raise KeyError(id)

    def text_content(self):
        """
        Return the text content of the tag (and the text in any children).
        """
        return _collect_string_content(self)

    def cssselect(self, expr, translator='html'):
        """
        Run the CSS expression on this element and its children,
        returning a list of the results.

        Equivalent to lxml.cssselect.CSSSelect(expr, translator='html')(self)
        -- note that pre-compiling the expression can provide a substantial
        speedup.
        """
        from lxml.cssselect import CSSSelector
        return CSSSelector(expr, translator=translator)(self)

    def make_links_absolute(self, base_url=None, resolve_base_href=True, handle_failures=None):
        """
        Make all links in the document absolute, given the
        ``base_url`` for the document (the full URL where the document
        came from), or if no ``base_url`` is given, then the ``.base_url``
        of the document.

        If ``resolve_base_href`` is true, then any ``<base href>``
        tags in the document are used *and* removed from the document.
        If it is false then any such tag is ignored.

        If ``handle_failures`` is None (default), a failure to process
        a URL will abort the processing.  If set to 'ignore', errors
        are ignored.  If set to 'discard', failing URLs will be removed.
        """
        if base_url is None:
            base_url = self.base_url
            if base_url is None:
                raise TypeError('No base_url given, and the document has no base_url')
        if resolve_base_href:
            self.resolve_base_href()
        if handle_failures == 'ignore':

            def link_repl(href):
                try:
                    return urljoin(base_url, href)
                except ValueError:
                    return href
        elif handle_failures == 'discard':

            def link_repl(href):
                try:
                    return urljoin(base_url, href)
                except ValueError:
                    return None
        elif handle_failures is None:

            def link_repl(href):
                return urljoin(base_url, href)
        else:
            raise ValueError('unexpected value for handle_failures: %r' % handle_failures)
        self.rewrite_links(link_repl)

    def resolve_base_href(self, handle_failures=None):
        """
        Find any ``<base href>`` tag in the document, and apply its
        values to all links found in the document.  Also remove the
        tag once it has been applied.

        If ``handle_failures`` is None (default), a failure to process
        a URL will abort the processing.  If set to 'ignore', errors
        are ignored.  If set to 'discard', failing URLs will be removed.
        """
        base_href = None
        basetags = self.xpath('//base[@href]|//x:base[@href]', namespaces={'x': XHTML_NAMESPACE})
        for b in basetags:
            base_href = b.get('href')
            b.drop_tree()
        if not base_href:
            return
        self.make_links_absolute(base_href, resolve_base_href=False, handle_failures=handle_failures)

    def iterlinks(self):
        """
        Yield (element, attribute, link, pos), where attribute may be None
        (indicating the link is in the text).  ``pos`` is the position
        where the link occurs; often 0, but sometimes something else in
        the case of links in stylesheets or style tags.

        Note: <base href> is *not* taken into account in any way.  The
        link you get is exactly the link in the document.

        Note: multiple links inside of a single text string or
        attribute value are returned in reversed order.  This makes it
        possible to replace or delete them from the text string value
        based on their reported text positions.  Otherwise, a
        modification at one text position can change the positions of
        links reported later on.
        """
        link_attrs = defs.link_attrs
        for el in self.iter(etree.Element):
            attribs = el.attrib
            tag = _nons(el.tag)
            if tag == 'object':
                codebase = None
                if 'codebase' in attribs:
                    codebase = el.get('codebase')
                    yield (el, 'codebase', codebase, 0)
                for attrib in ('classid', 'data'):
                    if attrib in attribs:
                        value = el.get(attrib)
                        if codebase is not None:
                            value = urljoin(codebase, value)
                        yield (el, attrib, value, 0)
                if 'archive' in attribs:
                    for match in _archive_re.finditer(el.get('archive')):
                        value = match.group(0)
                        if codebase is not None:
                            value = urljoin(codebase, value)
                        yield (el, 'archive', value, match.start())
            else:
                for attrib in link_attrs:
                    if attrib in attribs:
                        yield (el, attrib, attribs[attrib], 0)
            if tag == 'meta':
                http_equiv = attribs.get('http-equiv', '').lower()
                if http_equiv == 'refresh':
                    content = attribs.get('content', '')
                    match = _parse_meta_refresh_url(content)
                    url = (match.group('url') if match else content).strip()
                    if url:
                        url, pos = _unquote_match(url, match.start('url') if match else content.find(url))
                        yield (el, 'content', url, pos)
            elif tag == 'param':
                valuetype = el.get('valuetype') or ''
                if valuetype.lower() == 'ref':
                    yield (el, 'value', el.get('value'), 0)
            elif tag == 'style' and el.text:
                urls = [_unquote_match(match.group(1), match.start(1))[::-1] for match in _iter_css_urls(el.text)] + [(match.start(1), match.group(1)) for match in _iter_css_imports(el.text)]
                if urls:
                    urls.sort(reverse=True)
                    for start, url in urls:
                        yield (el, None, url, start)
            if 'style' in attribs:
                urls = list(_iter_css_urls(attribs['style']))
                if urls:
                    for match in urls[::-1]:
                        url, start = _unquote_match(match.group(1), match.start(1))
                        yield (el, 'style', url, start)

    def rewrite_links(self, link_repl_func, resolve_base_href=True, base_href=None):
        """
        Rewrite all the links in the document.  For each link
        ``link_repl_func(link)`` will be called, and the return value
        will replace the old link.

        Note that links may not be absolute (unless you first called
        ``make_links_absolute()``), and may be internal (e.g.,
        ``'#anchor'``).  They can also be values like
        ``'mailto:email'`` or ``'javascript:expr'``.

        If you give ``base_href`` then all links passed to
        ``link_repl_func()`` will take that into account.

        If the ``link_repl_func`` returns None, the attribute or
        tag text will be removed completely.
        """
        if base_href is not None:
            self.make_links_absolute(base_href, resolve_base_href=resolve_base_href)
        elif resolve_base_href:
            self.resolve_base_href()
        for el, attrib, link, pos in self.iterlinks():
            new_link = link_repl_func(link.strip())
            if new_link == link:
                continue
            if new_link is None:
                if attrib is None:
                    el.text = ''
                else:
                    del el.attrib[attrib]
                continue
            if attrib is None:
                new = el.text[:pos] + new_link + el.text[pos + len(link):]
                el.text = new
            else:
                cur = el.get(attrib)
                if not pos and len(cur) == len(link):
                    new = new_link
                else:
                    new = cur[:pos] + new_link + cur[pos + len(link):]
                el.set(attrib, new)