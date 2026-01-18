from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
class CSSMatch(_DocumentNav):
    """Perform CSS matching."""

    def __init__(self, selectors: ct.SelectorList, scope: bs4.Tag, namespaces: ct.Namespaces | None, flags: int) -> None:
        """Initialize."""
        self.assert_valid_input(scope)
        self.tag = scope
        self.cached_meta_lang = []
        self.cached_default_forms = []
        self.cached_indeterminate_forms = []
        self.selectors = selectors
        self.namespaces = {} if namespaces is None else namespaces
        self.flags = flags
        self.iframe_restrict = False
        doc = scope
        parent = self.get_parent(doc)
        while parent:
            doc = parent
            parent = self.get_parent(doc)
        root = None
        if not self.is_doc(doc):
            root = doc
        else:
            for child in self.get_children(doc):
                root = child
                break
        self.root = root
        self.scope = scope if scope is not doc else root
        self.has_html_namespace = self.has_html_ns(root)
        self.is_xml = self.is_xml_tree(doc)
        self.is_html = not self.is_xml or self.has_html_namespace

    def supports_namespaces(self) -> bool:
        """Check if namespaces are supported in the HTML type."""
        return self.is_xml or self.has_html_namespace

    def get_tag_ns(self, el: bs4.Tag) -> str:
        """Get tag namespace."""
        if self.supports_namespaces():
            namespace = ''
            ns = self.get_uri(el)
            if ns:
                namespace = ns
        else:
            namespace = NS_XHTML
        return namespace

    def is_html_tag(self, el: bs4.Tag) -> bool:
        """Check if tag is in HTML namespace."""
        return self.get_tag_ns(el) == NS_XHTML

    def get_tag(self, el: bs4.Tag) -> str | None:
        """Get tag."""
        name = self.get_tag_name(el)
        return util.lower(name) if name is not None and (not self.is_xml) else name

    def get_prefix(self, el: bs4.Tag) -> str | None:
        """Get prefix."""
        prefix = self.get_prefix_name(el)
        return util.lower(prefix) if prefix is not None and (not self.is_xml) else prefix

    def find_bidi(self, el: bs4.Tag) -> int | None:
        """Get directionality from element text."""
        for node in self.get_children(el, tags=False):
            if self.is_tag(node):
                direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(node, 'dir', '')), None)
                if self.get_tag(node) in ('bdi', 'script', 'style', 'textarea', 'iframe') or not self.is_html_tag(node) or direction is not None:
                    continue
                value = self.find_bidi(node)
                if value is not None:
                    return value
                continue
            if self.is_special_string(node):
                continue
            for c in node:
                bidi = unicodedata.bidirectional(c)
                if bidi in ('AL', 'R', 'L'):
                    return ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
        return None

    def extended_language_filter(self, lang_range: str, lang_tag: str) -> bool:
        """Filter the language tags."""
        match = True
        lang_range = RE_WILD_STRIP.sub('-', lang_range).lower()
        ranges = lang_range.split('-')
        subtags = lang_tag.lower().split('-')
        length = len(ranges)
        slength = len(subtags)
        rindex = 0
        sindex = 0
        r = ranges[rindex]
        s = subtags[sindex]
        if length == 1 and slength == 1 and (not r) and (r == s):
            return True
        if r != '*' and r != s or (r == '*' and slength == 1 and (not s)):
            match = False
        rindex += 1
        sindex += 1
        while match and rindex < length:
            r = ranges[rindex]
            try:
                s = subtags[sindex]
            except IndexError:
                match = False
                continue
            if not r:
                match = False
                continue
            elif s == r:
                rindex += 1
            elif len(s) == 1:
                match = False
                continue
            sindex += 1
        return match

    def match_attribute_name(self, el: bs4.Tag, attr: str, prefix: str | None) -> str | Sequence[str] | None:
        """Match attribute name and return value if it exists."""
        value = None
        if self.supports_namespaces():
            value = None
            if prefix:
                ns = self.namespaces.get(prefix)
                if ns is None and prefix != '*':
                    return None
            else:
                ns = None
            for k, v in self.iter_attributes(el):
                namespace, name = self.split_namespace(el, k)
                if ns is None:
                    if self.is_xml and attr == k or (not self.is_xml and util.lower(attr) == util.lower(k)):
                        value = v
                        break
                    continue
                if namespace is None or (ns != namespace and prefix != '*'):
                    continue
                if util.lower(attr) != util.lower(name) if not self.is_xml else attr != name:
                    continue
                value = v
                break
        else:
            for k, v in self.iter_attributes(el):
                if util.lower(attr) != util.lower(k):
                    continue
                value = v
                break
        return value

    def match_namespace(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
        """Match the namespace of the element."""
        match = True
        namespace = self.get_tag_ns(el)
        default_namespace = self.namespaces.get('')
        tag_ns = '' if tag.prefix is None else self.namespaces.get(tag.prefix)
        if tag.prefix is None and (default_namespace is not None and namespace != default_namespace):
            match = False
        elif tag.prefix is not None and tag.prefix == '' and namespace:
            match = False
        elif tag.prefix and tag.prefix != '*' and (tag_ns is None or namespace != tag_ns):
            match = False
        return match

    def match_attributes(self, el: bs4.Tag, attributes: tuple[ct.SelectorAttribute, ...]) -> bool:
        """Match attributes."""
        match = True
        if attributes:
            for a in attributes:
                temp = self.match_attribute_name(el, a.attribute, a.prefix)
                pattern = a.xml_type_pattern if self.is_xml and a.xml_type_pattern else a.pattern
                if temp is None:
                    match = False
                    break
                value = temp if isinstance(temp, str) else ' '.join(temp)
                if pattern is None:
                    continue
                elif pattern.match(value) is None:
                    match = False
                    break
        return match

    def match_tagname(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
        """Match tag name."""
        name = util.lower(tag.name) if not self.is_xml and tag.name is not None else tag.name
        return not (name is not None and name not in (self.get_tag(el), '*'))

    def match_tag(self, el: bs4.Tag, tag: ct.SelectorTag | None) -> bool:
        """Match the tag."""
        match = True
        if tag is not None:
            if not self.match_namespace(el, tag):
                match = False
            if not self.match_tagname(el, tag):
                match = False
        return match

    def match_past_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        """Match past relationship."""
        found = False
        if isinstance(relation[0], ct.SelectorNull):
            return found
        if relation[0].rel_type == REL_PARENT:
            parent = self.get_parent(el, no_iframe=self.iframe_restrict)
            while not found and parent:
                found = self.match_selectors(parent, relation)
                parent = self.get_parent(parent, no_iframe=self.iframe_restrict)
        elif relation[0].rel_type == REL_CLOSE_PARENT:
            parent = self.get_parent(el, no_iframe=self.iframe_restrict)
            if parent:
                found = self.match_selectors(parent, relation)
        elif relation[0].rel_type == REL_SIBLING:
            sibling = self.get_previous(el)
            while not found and sibling:
                found = self.match_selectors(sibling, relation)
                sibling = self.get_previous(sibling)
        elif relation[0].rel_type == REL_CLOSE_SIBLING:
            sibling = self.get_previous(el)
            if sibling and self.is_tag(sibling):
                found = self.match_selectors(sibling, relation)
        return found

    def match_future_child(self, parent: bs4.Tag, relation: ct.SelectorList, recursive: bool=False) -> bool:
        """Match future child."""
        match = False
        if recursive:
            children = self.get_descendants
        else:
            children = self.get_children
        for child in children(parent, no_iframe=self.iframe_restrict):
            match = self.match_selectors(child, relation)
            if match:
                break
        return match

    def match_future_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        """Match future relationship."""
        found = False
        if isinstance(relation[0], ct.SelectorNull):
            return found
        if relation[0].rel_type == REL_HAS_PARENT:
            found = self.match_future_child(el, relation, True)
        elif relation[0].rel_type == REL_HAS_CLOSE_PARENT:
            found = self.match_future_child(el, relation)
        elif relation[0].rel_type == REL_HAS_SIBLING:
            sibling = self.get_next(el)
            while not found and sibling:
                found = self.match_selectors(sibling, relation)
                sibling = self.get_next(sibling)
        elif relation[0].rel_type == REL_HAS_CLOSE_SIBLING:
            sibling = self.get_next(el)
            if sibling and self.is_tag(sibling):
                found = self.match_selectors(sibling, relation)
        return found

    def match_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        """Match relationship to other elements."""
        found = False
        if isinstance(relation[0], ct.SelectorNull) or relation[0].rel_type is None:
            return found
        if relation[0].rel_type.startswith(':'):
            found = self.match_future_relations(el, relation)
        else:
            found = self.match_past_relations(el, relation)
        return found

    def match_id(self, el: bs4.Tag, ids: tuple[str, ...]) -> bool:
        """Match element's ID."""
        found = True
        for i in ids:
            if i != self.get_attribute_by_name(el, 'id', ''):
                found = False
                break
        return found

    def match_classes(self, el: bs4.Tag, classes: tuple[str, ...]) -> bool:
        """Match element's classes."""
        current_classes = self.get_classes(el)
        found = True
        for c in classes:
            if c not in current_classes:
                found = False
                break
        return found

    def match_root(self, el: bs4.Tag) -> bool:
        """Match element as root."""
        is_root = self.is_root(el)
        if is_root:
            sibling = self.get_previous(el, tags=False)
            while is_root and sibling is not None:
                if self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or self.is_cdata(sibling):
                    is_root = False
                else:
                    sibling = self.get_previous(sibling, tags=False)
        if is_root:
            sibling = self.get_next(el, tags=False)
            while is_root and sibling is not None:
                if self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or self.is_cdata(sibling):
                    is_root = False
                else:
                    sibling = self.get_next(sibling, tags=False)
        return is_root

    def match_scope(self, el: bs4.Tag) -> bool:
        """Match element as scope."""
        return self.scope is el

    def match_nth_tag_type(self, el: bs4.Tag, child: bs4.Tag) -> bool:
        """Match tag type for `nth` matches."""
        return self.get_tag(child) == self.get_tag(el) and self.get_tag_ns(child) == self.get_tag_ns(el)

    def match_nth(self, el: bs4.Tag, nth: bs4.Tag) -> bool:
        """Match `nth` elements."""
        matched = True
        for n in nth:
            matched = False
            if n.selectors and (not self.match_selectors(el, n.selectors)):
                break
            parent = self.get_parent(el)
            if parent is None:
                parent = self.create_fake_parent(el)
            last = n.last
            last_index = len(parent) - 1
            index = last_index if last else 0
            relative_index = 0
            a = n.a
            b = n.b
            var = n.n
            count = 0
            count_incr = 1
            factor = -1 if last else 1
            idx = last_idx = a * count + b if var else a
            if var:
                adjust = None
                while idx < 1 or idx > last_index:
                    if idx < 0:
                        diff_low = 0 - idx
                        if adjust is not None and adjust == 1:
                            break
                        adjust = -1
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                        diff = 0 - idx
                        if diff >= diff_low:
                            break
                    else:
                        diff_high = idx - last_index
                        if adjust is not None and adjust == -1:
                            break
                        adjust = 1
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                        diff = idx - last_index
                        if diff >= diff_high:
                            break
                        diff_high = diff
                lowest = count
                if a < 0:
                    while idx >= 1:
                        lowest = count
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                    count_incr = -1
                count = lowest
                idx = last_idx = a * count + b if var else a
            while 1 <= idx <= last_index + 1:
                child = None
                for child in self.get_children(parent, start=index, reverse=factor < 0, tags=False):
                    index += factor
                    if not self.is_tag(child):
                        continue
                    if n.selectors and (not self.match_selectors(child, n.selectors)):
                        continue
                    if n.of_type and (not self.match_nth_tag_type(el, child)):
                        continue
                    relative_index += 1
                    if relative_index == idx:
                        if child is el:
                            matched = True
                        else:
                            break
                    if child is el:
                        break
                if child is el:
                    break
                last_idx = idx
                count += count_incr
                if count < 0:
                    break
                idx = a * count + b if var else a
                if last_idx == idx:
                    break
            if not matched:
                break
        return matched

    def match_empty(self, el: bs4.Tag) -> bool:
        """Check if element is empty (if requested)."""
        is_empty = True
        for child in self.get_children(el, tags=False):
            if self.is_tag(child):
                is_empty = False
                break
            elif self.is_content_string(child) and RE_NOT_EMPTY.search(child):
                is_empty = False
                break
        return is_empty

    def match_subselectors(self, el: bs4.Tag, selectors: tuple[ct.SelectorList, ...]) -> bool:
        """Match selectors."""
        match = True
        for sel in selectors:
            if not self.match_selectors(el, sel):
                match = False
        return match

    def match_contains(self, el: bs4.Tag, contains: tuple[ct.SelectorContains, ...]) -> bool:
        """Match element if it contains text."""
        match = True
        content = None
        for contain_list in contains:
            if content is None:
                if contain_list.own:
                    content = self.get_own_text(el, no_iframe=self.is_html)
                else:
                    content = self.get_text(el, no_iframe=self.is_html)
            found = False
            for text in contain_list.text:
                if contain_list.own:
                    for c in content:
                        if text in c:
                            found = True
                            break
                    if found:
                        break
                elif text in content:
                    found = True
                    break
            if not found:
                match = False
        return match

    def match_default(self, el: bs4.Tag) -> bool:
        """Match default."""
        match = False
        form = None
        parent = self.get_parent(el, no_iframe=True)
        while parent and form is None:
            if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                form = parent
            else:
                parent = self.get_parent(parent, no_iframe=True)
        found_form = False
        for f, t in self.cached_default_forms:
            if f is form:
                found_form = True
                if t is el:
                    match = True
                break
        if not found_form:
            for child in self.get_descendants(form, no_iframe=True):
                name = self.get_tag(child)
                if name == 'form':
                    break
                if name in ('input', 'button'):
                    v = self.get_attribute_by_name(child, 'type', '')
                    if v and util.lower(v) == 'submit':
                        self.cached_default_forms.append((form, child))
                        if el is child:
                            match = True
                        break
        return match

    def match_indeterminate(self, el: bs4.Tag) -> bool:
        """Match default."""
        match = False
        name = cast(str, self.get_attribute_by_name(el, 'name'))

        def get_parent_form(el: bs4.Tag) -> bs4.Tag | None:
            """Find this input's form."""
            form = None
            parent = self.get_parent(el, no_iframe=True)
            while form is None:
                if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                    form = parent
                    break
                last_parent = parent
                parent = self.get_parent(parent, no_iframe=True)
                if parent is None:
                    form = last_parent
                    break
            return form
        form = get_parent_form(el)
        found_form = False
        for f, n, i in self.cached_indeterminate_forms:
            if f is form and n == name:
                found_form = True
                if i is True:
                    match = True
                break
        if not found_form:
            checked = False
            for child in self.get_descendants(form, no_iframe=True):
                if child is el:
                    continue
                tag_name = self.get_tag(child)
                if tag_name == 'input':
                    is_radio = False
                    check = False
                    has_name = False
                    for k, v in self.iter_attributes(child):
                        if util.lower(k) == 'type' and util.lower(v) == 'radio':
                            is_radio = True
                        elif util.lower(k) == 'name' and v == name:
                            has_name = True
                        elif util.lower(k) == 'checked':
                            check = True
                        if is_radio and check and has_name and (get_parent_form(child) is form):
                            checked = True
                            break
                if checked:
                    break
            if not checked:
                match = True
            self.cached_indeterminate_forms.append((form, name, match))
        return match

    def match_lang(self, el: bs4.Tag, langs: tuple[ct.SelectorLang, ...]) -> bool:
        """Match languages."""
        match = False
        has_ns = self.supports_namespaces()
        root = self.root
        has_html_namespace = self.has_html_namespace
        parent = el
        found_lang = None
        last = None
        while not found_lang:
            has_html_ns = self.has_html_ns(parent)
            for k, v in self.iter_attributes(parent):
                attr_ns, attr = self.split_namespace(parent, k)
                if (not has_ns or has_html_ns) and (util.lower(k) if not self.is_xml else k) == 'lang' or (has_ns and (not has_html_ns) and (attr_ns == NS_XML) and ((util.lower(attr) if not self.is_xml and attr is not None else attr) == 'lang')):
                    found_lang = v
                    break
            last = parent
            parent = self.get_parent(parent, no_iframe=self.is_html)
            if parent is None:
                root = last
                has_html_namespace = self.has_html_ns(root)
                parent = last
                break
        if found_lang is None and self.cached_meta_lang:
            for cache in self.cached_meta_lang:
                if root is cache[0]:
                    found_lang = cache[1]
        if found_lang is None and (not self.is_xml or (has_html_namespace and root.name == 'html')):
            found = False
            for tag in ('html', 'head'):
                found = False
                for child in self.get_children(parent, no_iframe=self.is_html):
                    if self.get_tag(child) == tag and self.is_html_tag(child):
                        found = True
                        parent = child
                        break
                if not found:
                    break
            if found:
                for child in parent:
                    if self.is_tag(child) and self.get_tag(child) == 'meta' and self.is_html_tag(parent):
                        c_lang = False
                        content = None
                        for k, v in self.iter_attributes(child):
                            if util.lower(k) == 'http-equiv' and util.lower(v) == 'content-language':
                                c_lang = True
                            if util.lower(k) == 'content':
                                content = v
                            if c_lang and content:
                                found_lang = content
                                self.cached_meta_lang.append((cast(str, root), cast(str, found_lang)))
                                break
                    if found_lang is not None:
                        break
                if found_lang is None:
                    self.cached_meta_lang.append((cast(str, root), ''))
        if found_lang is not None:
            for patterns in langs:
                match = False
                for pattern in patterns:
                    if self.extended_language_filter(pattern, cast(str, found_lang)):
                        match = True
                if not match:
                    break
        return match

    def match_dir(self, el: bs4.Tag, directionality: int) -> bool:
        """Check directionality."""
        if directionality & ct.SEL_DIR_LTR and directionality & ct.SEL_DIR_RTL:
            return False
        if el is None or not self.is_html_tag(el):
            return False
        direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(el, 'dir', '')), None)
        if direction not in (None, 0):
            return direction == directionality
        is_root = self.is_root(el)
        if is_root and direction is None:
            return ct.SEL_DIR_LTR == directionality
        name = self.get_tag(el)
        is_input = name == 'input'
        is_textarea = name == 'textarea'
        is_bdi = name == 'bdi'
        itype = util.lower(self.get_attribute_by_name(el, 'type', '')) if is_input else ''
        if is_input and itype == 'tel' and (direction is None):
            return ct.SEL_DIR_LTR == directionality
        if (is_input and itype in ('text', 'search', 'tel', 'url', 'email') or is_textarea) and direction == 0:
            if is_textarea:
                value = ''.join((node for node in self.get_contents(el, no_iframe=True) if self.is_content_string(node)))
            else:
                value = cast(str, self.get_attribute_by_name(el, 'value', ''))
            if value:
                for c in value:
                    bidi = unicodedata.bidirectional(c)
                    if bidi in ('AL', 'R', 'L'):
                        direction = ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
                        return direction == directionality
                return ct.SEL_DIR_LTR == directionality
            elif is_root:
                return ct.SEL_DIR_LTR == directionality
            return self.match_dir(self.get_parent(el, no_iframe=True), directionality)
        if is_bdi and direction is None or direction == 0:
            direction = self.find_bidi(el)
            if direction is not None:
                return direction == directionality
            elif is_root:
                return ct.SEL_DIR_LTR == directionality
            return self.match_dir(self.get_parent(el, no_iframe=True), directionality)
        return self.match_dir(self.get_parent(el, no_iframe=True), directionality)

    def match_range(self, el: bs4.Tag, condition: int) -> bool:
        """
        Match range.

        Behavior is modeled after what we see in browsers. Browsers seem to evaluate
        if the value is out of range, and if not, it is in range. So a missing value
        will not evaluate out of range; therefore, value is in range. Personally, I
        feel like this should evaluate as neither in or out of range.
        """
        out_of_range = False
        itype = util.lower(self.get_attribute_by_name(el, 'type'))
        mn = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'min', None)))
        mx = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'max', None)))
        if mn is None and mx is None:
            return False
        value = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'value', None)))
        if value is not None:
            if itype in ('date', 'datetime-local', 'month', 'week', 'number', 'range'):
                if mn is not None and value < mn:
                    out_of_range = True
                if not out_of_range and mx is not None and (value > mx):
                    out_of_range = True
            elif itype == 'time':
                if mn is not None and mx is not None and (mn > mx):
                    if value < mn and value > mx:
                        out_of_range = True
                else:
                    if mn is not None and value < mn:
                        out_of_range = True
                    if not out_of_range and mx is not None and (value > mx):
                        out_of_range = True
        return not out_of_range if condition & ct.SEL_IN_RANGE else out_of_range

    def match_defined(self, el: bs4.Tag) -> bool:
        """
        Match defined.

        `:defined` is related to custom elements in a browser.

        - If the document is XML (not XHTML), all tags will match.
        - Tags that are not custom (don't have a hyphen) are marked defined.
        - If the tag has a prefix (without or without a namespace), it will not match.

        This is of course requires the parser to provide us with the proper prefix and namespace info,
        if it doesn't, there is nothing we can do.
        """
        name = self.get_tag(el)
        return name is not None and (name.find('-') == -1 or name.find(':') != -1 or self.get_prefix(el) is not None)

    def match_placeholder_shown(self, el: bs4.Tag) -> bool:
        """
        Match placeholder shown according to HTML spec.

        - text area should be checked if they have content. A single newline does not count as content.

        """
        match = False
        content = self.get_text(el)
        if content in ('', '\n'):
            match = True
        return match

    def match_selectors(self, el: bs4.Tag, selectors: ct.SelectorList) -> bool:
        """Check if element matches one of the selectors."""
        match = False
        is_not = selectors.is_not
        is_html = selectors.is_html
        if is_html:
            namespaces = self.namespaces
            iframe_restrict = self.iframe_restrict
            self.namespaces = {'html': NS_XHTML}
            self.iframe_restrict = True
        if not is_html or self.is_html:
            for selector in selectors:
                match = is_not
                if isinstance(selector, ct.SelectorNull):
                    continue
                if not self.match_tag(el, selector.tag):
                    continue
                if selector.flags & ct.SEL_DEFINED and (not self.match_defined(el)):
                    continue
                if selector.flags & ct.SEL_ROOT and (not self.match_root(el)):
                    continue
                if selector.flags & ct.SEL_SCOPE and (not self.match_scope(el)):
                    continue
                if selector.flags & ct.SEL_PLACEHOLDER_SHOWN and (not self.match_placeholder_shown(el)):
                    continue
                if not self.match_nth(el, selector.nth):
                    continue
                if selector.flags & ct.SEL_EMPTY and (not self.match_empty(el)):
                    continue
                if selector.ids and (not self.match_id(el, selector.ids)):
                    continue
                if selector.classes and (not self.match_classes(el, selector.classes)):
                    continue
                if not self.match_attributes(el, selector.attributes):
                    continue
                if selector.flags & RANGES and (not self.match_range(el, selector.flags & RANGES)):
                    continue
                if selector.lang and (not self.match_lang(el, selector.lang)):
                    continue
                if selector.selectors and (not self.match_subselectors(el, selector.selectors)):
                    continue
                if selector.relation and (not self.match_relations(el, selector.relation)):
                    continue
                if selector.flags & ct.SEL_DEFAULT and (not self.match_default(el)):
                    continue
                if selector.flags & ct.SEL_INDETERMINATE and (not self.match_indeterminate(el)):
                    continue
                if selector.flags & DIR_FLAGS and (not self.match_dir(el, selector.flags & DIR_FLAGS)):
                    continue
                if selector.contains and (not self.match_contains(el, selector.contains)):
                    continue
                match = not is_not
                break
        if is_html:
            self.namespaces = namespaces
            self.iframe_restrict = iframe_restrict
        return match

    def select(self, limit: int=0) -> Iterator[bs4.Tag]:
        """Match all tags under the targeted tag."""
        lim = None if limit < 1 else limit
        for child in self.get_descendants(self.tag):
            if self.match(child):
                yield child
                if lim is not None:
                    lim -= 1
                    if lim < 1:
                        break

    def closest(self) -> bs4.Tag | None:
        """Match closest ancestor."""
        current = self.tag
        closest = None
        while closest is None and current is not None:
            if self.match(current):
                closest = current
            else:
                current = self.get_parent(current)
        return closest

    def filter(self) -> list[bs4.Tag]:
        """Filter tag's children."""
        return [tag for tag in self.get_contents(self.tag) if not self.is_navigable_string(tag) and self.match(tag)]

    def match(self, el: bs4.Tag) -> bool:
        """Match."""
        return not self.is_doc(el) and self.is_tag(el) and self.match_selectors(el, self.selectors)