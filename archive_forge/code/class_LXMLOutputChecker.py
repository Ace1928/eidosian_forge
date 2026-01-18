from lxml import etree
import sys
import re
import doctest
class LXMLOutputChecker(OutputChecker):
    empty_tags = ('param', 'img', 'area', 'br', 'basefont', 'input', 'base', 'meta', 'link', 'col')

    def get_default_parser(self):
        return etree.XML

    def check_output(self, want, got, optionflags):
        alt_self = getattr(self, '_temp_override_self', None)
        if alt_self is not None:
            super_method = self._temp_call_super_check_output
            self = alt_self
        else:
            super_method = OutputChecker.check_output
        parser = self.get_parser(want, got, optionflags)
        if not parser:
            return super_method(self, want, got, optionflags)
        try:
            want_doc = parser(want)
        except etree.XMLSyntaxError:
            return False
        try:
            got_doc = parser(got)
        except etree.XMLSyntaxError:
            return False
        return self.compare_docs(want_doc, got_doc)

    def get_parser(self, want, got, optionflags):
        parser = None
        if NOPARSE_MARKUP & optionflags:
            return None
        if PARSE_HTML & optionflags:
            parser = html_fromstring
        elif PARSE_XML & optionflags:
            parser = etree.XML
        elif want.strip().lower().startswith('<html') and got.strip().startswith('<html'):
            parser = html_fromstring
        elif self._looks_like_markup(want) and self._looks_like_markup(got):
            parser = self.get_default_parser()
        return parser

    def _looks_like_markup(self, s):
        s = s.strip()
        return s.startswith('<') and (not _repr_re.search(s))

    def compare_docs(self, want, got):
        if not self.tag_compare(want.tag, got.tag):
            return False
        if not self.text_compare(want.text, got.text, True):
            return False
        if not self.text_compare(want.tail, got.tail, True):
            return False
        if 'any' not in want.attrib:
            want_keys = sorted(want.attrib.keys())
            got_keys = sorted(got.attrib.keys())
            if want_keys != got_keys:
                return False
            for key in want_keys:
                if not self.text_compare(want.attrib[key], got.attrib[key], False):
                    return False
        if want.text != '...' or len(want):
            want_children = list(want)
            got_children = list(got)
            while want_children or got_children:
                if not want_children or not got_children:
                    return False
                want_first = want_children.pop(0)
                got_first = got_children.pop(0)
                if not self.compare_docs(want_first, got_first):
                    return False
                if not got_children and want_first.tail == '...':
                    break
        return True

    def text_compare(self, want, got, strip):
        want = want or ''
        got = got or ''
        if strip:
            want = norm_whitespace(want).strip()
            got = norm_whitespace(got).strip()
        want = '^%s$' % re.escape(want)
        want = want.replace('\\.\\.\\.', '.*')
        if re.search(want, got):
            return True
        else:
            return False

    def tag_compare(self, want, got):
        if want == 'any':
            return True
        if not isinstance(want, (str, bytes)) or not isinstance(got, (str, bytes)):
            return want == got
        want = want or ''
        got = got or ''
        if want.startswith('{...}'):
            return want.split('}')[-1] == got.split('}')[-1]
        else:
            return want == got

    def output_difference(self, example, got, optionflags):
        want = example.want
        parser = self.get_parser(want, got, optionflags)
        errors = []
        if parser is not None:
            try:
                want_doc = parser(want)
            except etree.XMLSyntaxError:
                e = sys.exc_info()[1]
                errors.append('In example: %s' % e)
            try:
                got_doc = parser(got)
            except etree.XMLSyntaxError:
                e = sys.exc_info()[1]
                errors.append('In actual output: %s' % e)
        if parser is None or errors:
            value = OutputChecker.output_difference(self, example, got, optionflags)
            if errors:
                errors.append(value)
                return '\n'.join(errors)
            else:
                return value
        html = parser is html_fromstring
        diff_parts = ['Expected:', self.format_doc(want_doc, html, 2), 'Got:', self.format_doc(got_doc, html, 2), 'Diff:', self.collect_diff(want_doc, got_doc, html, 2)]
        return '\n'.join(diff_parts)

    def html_empty_tag(self, el, html=True):
        if not html:
            return False
        if el.tag not in self.empty_tags:
            return False
        if el.text or len(el):
            return False
        return True

    def format_doc(self, doc, html, indent, prefix=''):
        parts = []
        if not len(doc):
            parts.append(' ' * indent)
            parts.append(prefix)
            parts.append(self.format_tag(doc))
            if not self.html_empty_tag(doc, html):
                if strip(doc.text):
                    parts.append(self.format_text(doc.text))
                parts.append(self.format_end_tag(doc))
            if strip(doc.tail):
                parts.append(self.format_text(doc.tail))
            parts.append('\n')
            return ''.join(parts)
        parts.append(' ' * indent)
        parts.append(prefix)
        parts.append(self.format_tag(doc))
        if not self.html_empty_tag(doc, html):
            parts.append('\n')
            if strip(doc.text):
                parts.append(' ' * indent)
                parts.append(self.format_text(doc.text))
                parts.append('\n')
            for el in doc:
                parts.append(self.format_doc(el, html, indent + 2))
            parts.append(' ' * indent)
            parts.append(self.format_end_tag(doc))
            parts.append('\n')
        if strip(doc.tail):
            parts.append(' ' * indent)
            parts.append(self.format_text(doc.tail))
            parts.append('\n')
        return ''.join(parts)

    def format_text(self, text, strip=True):
        if text is None:
            return ''
        if strip:
            text = text.strip()
        return html_escape(text, 1)

    def format_tag(self, el):
        attrs = []
        if isinstance(el, etree.CommentBase):
            return '<!--'
        for name, value in sorted(el.attrib.items()):
            attrs.append('%s="%s"' % (name, self.format_text(value, False)))
        if not attrs:
            return '<%s>' % el.tag
        return '<%s %s>' % (el.tag, ' '.join(attrs))

    def format_end_tag(self, el):
        if isinstance(el, etree.CommentBase):
            return '-->'
        return '</%s>' % el.tag

    def collect_diff(self, want, got, html, indent):
        parts = []
        if not len(want) and (not len(got)):
            parts.append(' ' * indent)
            parts.append(self.collect_diff_tag(want, got))
            if not self.html_empty_tag(got, html):
                parts.append(self.collect_diff_text(want.text, got.text))
                parts.append(self.collect_diff_end_tag(want, got))
            parts.append(self.collect_diff_text(want.tail, got.tail))
            parts.append('\n')
            return ''.join(parts)
        parts.append(' ' * indent)
        parts.append(self.collect_diff_tag(want, got))
        parts.append('\n')
        if strip(want.text) or strip(got.text):
            parts.append(' ' * indent)
            parts.append(self.collect_diff_text(want.text, got.text))
            parts.append('\n')
        want_children = list(want)
        got_children = list(got)
        while want_children or got_children:
            if not want_children:
                parts.append(self.format_doc(got_children.pop(0), html, indent + 2, '+'))
                continue
            if not got_children:
                parts.append(self.format_doc(want_children.pop(0), html, indent + 2, '-'))
                continue
            parts.append(self.collect_diff(want_children.pop(0), got_children.pop(0), html, indent + 2))
        parts.append(' ' * indent)
        parts.append(self.collect_diff_end_tag(want, got))
        parts.append('\n')
        if strip(want.tail) or strip(got.tail):
            parts.append(' ' * indent)
            parts.append(self.collect_diff_text(want.tail, got.tail))
            parts.append('\n')
        return ''.join(parts)

    def collect_diff_tag(self, want, got):
        if not self.tag_compare(want.tag, got.tag):
            tag = '%s (got: %s)' % (want.tag, got.tag)
        else:
            tag = got.tag
        attrs = []
        any = want.tag == 'any' or 'any' in want.attrib
        for name, value in sorted(got.attrib.items()):
            if name not in want.attrib and (not any):
                attrs.append('+%s="%s"' % (name, self.format_text(value, False)))
            else:
                if name in want.attrib:
                    text = self.collect_diff_text(want.attrib[name], value, False)
                else:
                    text = self.format_text(value, False)
                attrs.append('%s="%s"' % (name, text))
        if not any:
            for name, value in sorted(want.attrib.items()):
                if name in got.attrib:
                    continue
                attrs.append('-%s="%s"' % (name, self.format_text(value, False)))
        if attrs:
            tag = '<%s %s>' % (tag, ' '.join(attrs))
        else:
            tag = '<%s>' % tag
        return tag

    def collect_diff_end_tag(self, want, got):
        if want.tag != got.tag:
            tag = '%s (got: %s)' % (want.tag, got.tag)
        else:
            tag = got.tag
        return '</%s>' % tag

    def collect_diff_text(self, want, got, strip=True):
        if self.text_compare(want, got, strip):
            if not got:
                return ''
            return self.format_text(got, strip)
        text = '%s (got: %s)' % (want, got)
        return self.format_text(text, strip)