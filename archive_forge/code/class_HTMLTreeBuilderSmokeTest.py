import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
class HTMLTreeBuilderSmokeTest(TreeBuilderSmokeTest):
    """A basic test of a treebuilder's competence.

    Any HTML treebuilder, present or future, should be able to pass
    these tests. With invalid markup, there's room for interpretation,
    and different parsers can handle it differently. But with the
    markup in these tests, there's not much room for interpretation.
    """

    def test_empty_element_tags(self):
        """Verify that all HTML4 and HTML5 empty element (aka void element) tags
        are handled correctly.
        """
        for name in ['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'keygen', 'link', 'menuitem', 'meta', 'param', 'source', 'track', 'wbr', 'spacer', 'frame']:
            soup = self.soup('')
            new_tag = soup.new_tag(name)
            assert new_tag.is_empty_element == True

    def test_special_string_containers(self):
        soup = self.soup('<style>Some CSS</style><script>Some Javascript</script>')
        assert isinstance(soup.style.string, Stylesheet)
        assert isinstance(soup.script.string, Script)
        soup = self.soup('<style><!--Some CSS--></style>')
        assert isinstance(soup.style.string, Stylesheet)
        assert soup.style.string == '<!--Some CSS-->'
        assert isinstance(soup.style.string, Stylesheet)

    def test_pickle_and_unpickle_identity(self):
        tree = self.soup('<a><b>foo</a>')
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == tree.decode()

    def assertDoctypeHandled(self, doctype_fragment):
        """Assert that a given doctype string is handled correctly."""
        doctype_str, soup = self._document_with_doctype(doctype_fragment)
        doctype = soup.contents[0]
        assert doctype.__class__ == Doctype
        assert doctype == doctype_fragment
        assert soup.encode('utf8')[:len(doctype_str)] == doctype_str
        assert soup.p.contents[0] == 'foo'

    def _document_with_doctype(self, doctype_fragment, doctype_string='DOCTYPE'):
        """Generate and parse a document with the given doctype."""
        doctype = '<!%s %s>' % (doctype_string, doctype_fragment)
        markup = doctype + '\n<p>foo</p>'
        soup = self.soup(markup)
        return (doctype.encode('utf8'), soup)

    def test_normal_doctypes(self):
        """Make sure normal, everyday HTML doctypes are handled correctly."""
        self.assertDoctypeHandled('html')
        self.assertDoctypeHandled('html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"')

    def test_empty_doctype(self):
        soup = self.soup('<!DOCTYPE>')
        doctype = soup.contents[0]
        assert '' == doctype.strip()

    def test_mixed_case_doctype(self):
        for doctype_fragment in ('doctype', 'DocType'):
            doctype_str, soup = self._document_with_doctype('html', doctype_fragment)
            doctype = soup.contents[0]
            assert doctype.__class__ == Doctype
            assert doctype == 'html'
            assert soup.encode('utf8')[:len(doctype_str)] == b'<!DOCTYPE html>'
            assert soup.p.contents[0] == 'foo'

    def test_public_doctype_with_url(self):
        doctype = 'html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"'
        self.assertDoctypeHandled(doctype)

    def test_system_doctype(self):
        self.assertDoctypeHandled('foo SYSTEM "http://www.example.com/"')

    def test_namespaced_system_doctype(self):
        self.assertDoctypeHandled('xsl:stylesheet SYSTEM "htmlent.dtd"')

    def test_namespaced_public_doctype(self):
        self.assertDoctypeHandled('xsl:stylesheet PUBLIC "htmlent.dtd"')

    def test_real_xhtml_document(self):
        """A real XHTML document should come out more or less the same as it went in."""
        markup = b'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">\n<html xmlns="http://www.w3.org/1999/xhtml">\n<head><title>Hello.</title></head>\n<body>Goodbye.</body>\n</html>'
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
        assert soup.encode('utf-8').replace(b'\n', b'') == markup.replace(b'\n', b'')
        assert w == []

    def test_namespaced_html(self):
        markup = b'<ns1:foo>content</ns1:foo><ns1:foo/><ns2:foo/>'
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
        assert 2 == len(soup.find_all('ns1:foo'))
        assert [] == w

    def test_detect_xml_parsed_as_html(self):
        markup = b'<?xml version="1.0" encoding="utf-8"?><tag>string</tag>'
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
            assert soup.tag.string == 'string'
        [warning] = w
        assert isinstance(warning.message, XMLParsedAsHTMLWarning)
        assert str(warning.message) == XMLParsedAsHTMLWarning.MESSAGE

    def test_processing_instruction(self):
        markup = '<?PITarget PIContent?>'
        soup = self.soup(markup)
        assert markup == soup.decode()
        markup = b'<?PITarget PIContent?>'
        soup = self.soup(markup)
        assert markup == soup.encode('utf8')

    def test_deepcopy(self):
        """Make sure you can copy the tree builder.

        This is important because the builder is part of a
        BeautifulSoup object, and we want to be able to copy that.
        """
        copy.deepcopy(self.default_builder)

    def test_p_tag_is_never_empty_element(self):
        """A <p> tag is never designated as an empty-element tag.

        Even if the markup shows it as an empty-element tag, it
        shouldn't be presented that way.
        """
        soup = self.soup('<p/>')
        assert not soup.p.is_empty_element
        assert str(soup.p) == '<p></p>'

    def test_unclosed_tags_get_closed(self):
        """A tag that's not closed by the end of the document should be closed.

        This applies to all tags except empty-element tags.
        """
        self.assert_soup('<p>', '<p></p>')
        self.assert_soup('<b>', '<b></b>')
        self.assert_soup('<br>', '<br/>')

    def test_br_is_always_empty_element_tag(self):
        """A <br> tag is designated as an empty-element tag.

        Some parsers treat <br></br> as one <br/> tag, some parsers as
        two tags, but it should always be an empty-element tag.
        """
        soup = self.soup('<br></br>')
        assert soup.br.is_empty_element
        assert str(soup.br) == '<br/>'

    def test_nested_formatting_elements(self):
        self.assert_soup('<em><em></em></em>')

    def test_double_head(self):
        html = '<!DOCTYPE html>\n<html>\n<head>\n<title>Ordinary HEAD element test</title>\n</head>\n<script type="text/javascript">\nalert("Help!");\n</script>\n<body>\nHello, world!\n</body>\n</html>\n'
        soup = self.soup(html)
        assert 'text/javascript' == soup.find('script')['type']

    def test_comment(self):
        markup = '<p>foo<!--foobar-->baz</p>'
        self.assert_soup(markup)
        soup = self.soup(markup)
        comment = soup.find(string='foobar')
        assert comment.__class__ == Comment
        foo = soup.find(string='foo')
        assert comment == foo.next_element
        baz = soup.find(string='baz')
        assert comment == baz.previous_element

    def test_preserved_whitespace_in_pre_and_textarea(self):
        """Whitespace must be preserved in <pre> and <textarea> tags,
        even if that would mean not prettifying the markup.
        """
        pre_markup = '<pre>a   z</pre>\n'
        textarea_markup = '<textarea> woo\nwoo  </textarea>\n'
        self.assert_soup(pre_markup)
        self.assert_soup(textarea_markup)
        soup = self.soup(pre_markup)
        assert soup.pre.prettify() == pre_markup
        soup = self.soup(textarea_markup)
        assert soup.textarea.prettify() == textarea_markup
        soup = self.soup('<textarea></textarea>')
        assert soup.textarea.prettify() == '<textarea></textarea>\n'

    def test_nested_inline_elements(self):
        """Inline elements can be nested indefinitely."""
        b_tag = '<b>Inside a B tag</b>'
        self.assert_soup(b_tag)
        nested_b_tag = '<p>A <i>nested <b>tag</b></i></p>'
        self.assert_soup(nested_b_tag)
        double_nested_b_tag = '<p>A <a>doubly <i>nested <b>tag</b></i></a></p>'
        self.assert_soup(nested_b_tag)

    def test_nested_block_level_elements(self):
        """Block elements can be nested."""
        soup = self.soup('<blockquote><p><b>Foo</b></p></blockquote>')
        blockquote = soup.blockquote
        assert blockquote.p.b.string == 'Foo'
        assert blockquote.b.string == 'Foo'

    def test_correctly_nested_tables(self):
        """One table can go inside another one."""
        markup = '<table id="1"><tr><td>Here\'s another table:<table id="2"><tr><td>foo</td></tr></table></td>'
        self.assert_soup(markup, '<table id="1"><tr><td>Here\'s another table:<table id="2"><tr><td>foo</td></tr></table></td></tr></table>')
        self.assert_soup('<table><thead><tr><td>Foo</td></tr></thead><tbody><tr><td>Bar</td></tr></tbody><tfoot><tr><td>Baz</td></tr></tfoot></table>')

    def test_multivalued_attribute_with_whitespace(self):
        markup = '<div class=" foo bar\t "></a>'
        soup = self.soup(markup)
        assert ['foo', 'bar'] == soup.div['class']
        assert soup.div == soup.find('div', class_='foo bar')

    def test_deeply_nested_multivalued_attribute(self):
        markup = '<table><div><div class="css"></div></div></table>'
        soup = self.soup(markup)
        assert ['css'] == soup.div.div['class']

    def test_multivalued_attribute_on_html(self):
        markup = '<html class="a b"></html>'
        soup = self.soup(markup)
        assert ['a', 'b'] == soup.html['class']

    def test_angle_brackets_in_attribute_values_are_escaped(self):
        self.assert_soup('<a b="<a>"></a>', '<a b="&lt;a&gt;"></a>')

    def test_strings_resembling_character_entity_references(self):
        self.assert_soup('<p>&bull; AT&T is in the s&p 500</p>', '<p>• AT&amp;T is in the s&amp;p 500</p>')

    def test_apos_entity(self):
        self.assert_soup('<p>Bob&apos;s Bar</p>', "<p>Bob's Bar</p>")

    def test_entities_in_foreign_document_encoding(self):
        markup = '<p>&#147;Hello&#148; &#45;&#9731;</p>'
        soup = self.soup(markup)
        assert '“Hello” -☃' == soup.p.string

    def test_entities_in_attributes_converted_to_unicode(self):
        expect = '<p id="piñata"></p>'
        self.assert_soup('<p id="pi&#241;ata"></p>', expect)
        self.assert_soup('<p id="pi&#xf1;ata"></p>', expect)
        self.assert_soup('<p id="pi&#Xf1;ata"></p>', expect)
        self.assert_soup('<p id="pi&ntilde;ata"></p>', expect)

    def test_entities_in_text_converted_to_unicode(self):
        expect = '<p>piñata</p>'
        self.assert_soup('<p>pi&#241;ata</p>', expect)
        self.assert_soup('<p>pi&#xf1;ata</p>', expect)
        self.assert_soup('<p>pi&#Xf1;ata</p>', expect)
        self.assert_soup('<p>pi&ntilde;ata</p>', expect)

    def test_quot_entity_converted_to_quotation_mark(self):
        self.assert_soup('<p>I said &quot;good day!&quot;</p>', '<p>I said "good day!"</p>')

    def test_out_of_range_entity(self):
        expect = '�'
        self.assert_soup('&#10000000000000;', expect)
        self.assert_soup('&#x10000000000000;', expect)
        self.assert_soup('&#1000000000;', expect)

    def test_multipart_strings(self):
        """Mostly to prevent a recurrence of a bug in the html5lib treebuilder."""
        soup = self.soup('<html><h2>\nfoo</h2><p></p></html>')
        assert 'p' == soup.h2.string.next_element.name
        assert 'p' == soup.p.name
        self.assertConnectedness(soup)

    def test_empty_element_tags(self):
        """Verify consistent handling of empty-element tags,
        no matter how they come in through the markup.
        """
        self.assert_soup('<br/><br/><br/>', '<br/><br/><br/>')
        self.assert_soup('<br /><br /><br />', '<br/><br/><br/>')

    def test_head_tag_between_head_and_body(self):
        """Prevent recurrence of a bug in the html5lib treebuilder."""
        content = '<html><head></head>\n  <link></link>\n  <body>foo</body>\n</html>\n'
        soup = self.soup(content)
        assert soup.html.body is not None
        self.assertConnectedness(soup)

    def test_multiple_copies_of_a_tag(self):
        """Prevent recurrence of a bug in the html5lib treebuilder."""
        content = '<!DOCTYPE html>\n<html>\n <body>\n   <article id="a" >\n   <div><a href="1"></div>\n   <footer>\n     <a href="2"></a>\n   </footer>\n  </article>\n  </body>\n</html>\n'
        soup = self.soup(content)
        self.assertConnectedness(soup.article)

    def test_basic_namespaces(self):
        """Parsers don't need to *understand* namespaces, but at the
        very least they should not choke on namespaces or lose
        data."""
        markup = b'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:mathml="http://www.w3.org/1998/Math/MathML" xmlns:svg="http://www.w3.org/2000/svg"><head></head><body><mathml:msqrt>4</mathml:msqrt><b svg:fill="red"></b></body></html>'
        soup = self.soup(markup)
        assert markup == soup.encode()
        html = soup.html
        assert 'http://www.w3.org/1999/xhtml' == soup.html['xmlns']
        assert 'http://www.w3.org/1998/Math/MathML' == soup.html['xmlns:mathml']
        assert 'http://www.w3.org/2000/svg' == soup.html['xmlns:svg']

    def test_multivalued_attribute_value_becomes_list(self):
        markup = b'<a class="foo bar">'
        soup = self.soup(markup)
        assert ['foo', 'bar'] == soup.a['class']

    def test_can_parse_unicode_document(self):
        markup = '<html><head><meta encoding="euc-jp"></head><body>Sacré bleu!</body>'
        soup = self.soup(markup)
        assert 'Sacré bleu!' == soup.body.string

    def test_soupstrainer(self):
        """Parsers should be able to work with SoupStrainers."""
        strainer = SoupStrainer('b')
        soup = self.soup('A <b>bold</b> <meta/> <i>statement</i>', parse_only=strainer)
        assert soup.decode() == '<b>bold</b>'

    def test_single_quote_attribute_values_become_double_quotes(self):
        self.assert_soup("<foo attr='bar'></foo>", '<foo attr="bar"></foo>')

    def test_attribute_values_with_nested_quotes_are_left_alone(self):
        text = '<foo attr=\'bar "brawls" happen\'>a</foo>'
        self.assert_soup(text)

    def test_attribute_values_with_double_nested_quotes_get_quoted(self):
        text = '<foo attr=\'bar "brawls" happen\'>a</foo>'
        soup = self.soup(text)
        soup.foo['attr'] = 'Brawls happen at "Bob\'s Bar"'
        self.assert_soup(soup.foo.decode(), '<foo attr="Brawls happen at &quot;Bob\'s Bar&quot;">a</foo>')

    def test_ampersand_in_attribute_value_gets_escaped(self):
        self.assert_soup('<this is="really messed up & stuff"></this>', '<this is="really messed up &amp; stuff"></this>')
        self.assert_soup('<a href="http://example.org?a=1&b=2;3">foo</a>', '<a href="http://example.org?a=1&amp;b=2;3">foo</a>')

    def test_escaped_ampersand_in_attribute_value_is_left_alone(self):
        self.assert_soup('<a href="http://example.org?a=1&amp;b=2;3"></a>')

    def test_entities_in_strings_converted_during_parsing(self):
        text = '<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>'
        expected = '<p>&lt;&lt;sacré bleu!&gt;&gt;</p>'
        self.assert_soup(text, expected)

    def test_smart_quotes_converted_on_the_way_in(self):
        quote = b'<p>\x91Foo\x92</p>'
        soup = self.soup(quote)
        assert soup.p.string == '‘Foo’'

    def test_non_breaking_spaces_converted_on_the_way_in(self):
        soup = self.soup('<a>&nbsp;&nbsp;</a>')
        assert soup.a.string == '\xa0' * 2

    def test_entities_converted_on_the_way_out(self):
        text = '<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>'
        expected = '<p>&lt;&lt;sacré bleu!&gt;&gt;</p>'.encode('utf-8')
        soup = self.soup(text)
        assert soup.p.encode('utf-8') == expected

    def test_real_iso_8859_document(self):
        unicode_html = '<html><head><meta content="text/html; charset=ISO-8859-1" http-equiv="Content-type"/></head><body><p>Sacré bleu!</p></body></html>'
        iso_latin_html = unicode_html.encode('iso-8859-1')
        soup = self.soup(iso_latin_html)
        result = soup.encode('utf-8')
        expected = unicode_html.replace('ISO-8859-1', 'utf-8')
        expected = expected.encode('utf-8')
        assert result == expected

    def test_real_shift_jis_document(self):
        shift_jis_html = b'<html><head></head><body><pre>\x82\xb1\x82\xea\x82\xcdShift-JIS\x82\xc5\x83R\x81[\x83f\x83B\x83\x93\x83O\x82\xb3\x82\xea\x82\xbd\x93\xfa\x96{\x8c\xea\x82\xcc\x83t\x83@\x83C\x83\x8b\x82\xc5\x82\xb7\x81B</pre></body></html>'
        unicode_html = shift_jis_html.decode('shift-jis')
        soup = self.soup(unicode_html)
        assert soup.encode('utf-8') == unicode_html.encode('utf-8')
        assert soup.encode('euc_jp') == unicode_html.encode('euc_jp')

    def test_real_hebrew_document(self):
        hebrew_document = b'<html><head><title>Hebrew (ISO 8859-8) in Visual Directionality</title></head><body><h1>Hebrew (ISO 8859-8) in Visual Directionality</h1>\xed\xe5\xec\xf9</body></html>'
        soup = self.soup(hebrew_document, from_encoding='iso8859-8')
        assert soup.original_encoding in ('iso8859-8', 'iso-8859-8')
        assert soup.encode('utf-8') == hebrew_document.decode('iso8859-8').encode('utf-8')

    def test_meta_tag_reflects_current_encoding(self):
        meta_tag = '<meta content="text/html; charset=x-sjis" http-equiv="Content-type"/>'
        shift_jis_html = '<html><head>\n%s\n<meta http-equiv="Content-language" content="ja"/></head><body>Shift-JIS markup goes here.' % meta_tag
        soup = self.soup(shift_jis_html)
        parsed_meta = soup.find('meta', {'http-equiv': 'Content-type'})
        content = parsed_meta['content']
        assert 'text/html; charset=x-sjis' == content
        assert isinstance(content, ContentMetaAttributeValue)
        assert 'text/html; charset=utf8' == content.encode('utf8')

    def test_html5_style_meta_tag_reflects_current_encoding(self):
        meta_tag = '<meta id="encoding" charset="x-sjis" />'
        shift_jis_html = '<html><head>\n%s\n<meta http-equiv="Content-language" content="ja"/></head><body>Shift-JIS markup goes here.' % meta_tag
        soup = self.soup(shift_jis_html)
        parsed_meta = soup.find('meta', id='encoding')
        charset = parsed_meta['charset']
        assert 'x-sjis' == charset
        assert isinstance(charset, CharsetMetaAttributeValue)
        assert 'utf8' == charset.encode('utf8')

    def test_python_specific_encodings_not_used_in_charset(self):
        for markup in [b'<meta charset="utf8"></head><meta id="encoding" charset="utf-8" />']:
            soup = self.soup(markup)
            for encoding in PYTHON_SPECIFIC_ENCODINGS:
                if encoding in ('idna', 'mbcs', 'oem', 'undefined', 'string_escape', 'string-escape'):
                    continue
                encoded = soup.encode(encoding)
                assert b'meta charset=""' in encoded
                assert encoding.encode('ascii') not in encoded

    def test_tag_with_no_attributes_can_have_attributes_added(self):
        data = self.soup('<a>text</a>')
        data.a['foo'] = 'bar'
        assert '<a foo="bar">text</a>' == data.a.decode()

    def test_closing_tag_with_no_opening_tag(self):
        soup = self.soup('<body><div><p>text1</p></span>text2</div></body>')
        assert '<body><div><p>text1</p>text2</div></body>' == soup.body.decode()

    def test_worst_case(self):
        """Test the worst case (currently) for linking issues."""
        soup = self.soup(BAD_DOCUMENT)
        self.linkage_validator(soup)