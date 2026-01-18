from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest
class TestHTMLParserTreeBuilder(SoupTest, HTMLTreeBuilderSmokeTest):
    default_builder = HTMLParserTreeBuilder

    def test_rejected_input(self):
        bad_markup = [b'\n<![\xff\xfe\xfe\xcd\x00', b'<![n\x00', b'<![UNKNOWN[]]>']
        for markup in bad_markup:
            with pytest.raises(ParserRejectedMarkup):
                soup = self.soup(markup)

    def test_namespaced_system_doctype(self):
        pass

    def test_namespaced_public_doctype(self):
        pass

    def test_builder_is_pickled(self):
        """Unlike most tree builders, HTMLParserTreeBuilder and will
        be restored after pickling.
        """
        tree = self.soup('<a><b>foo</a>')
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert isinstance(loaded.builder, type(tree.builder))

    def test_redundant_empty_element_closing_tags(self):
        self.assert_soup('<br></br><br></br><br></br>', '<br/><br/><br/>')
        self.assert_soup('</br></br></br>', '')

    def test_empty_element(self):
        self.assert_soup('foo &# bar', 'foo &amp;# bar')

    def test_tracking_line_numbers(self):
        markup = '\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>'
        soup = self.soup(markup)
        assert 2 == soup.p.sourceline
        assert 3 == soup.p.sourcepos
        assert 'sourceline' == soup.p.find('sourceline').name
        soup = self.soup(markup, store_line_numbers=False)
        assert 'sourceline' == soup.p.sourceline.name
        assert 'sourcepos' == soup.p.sourcepos.name

    def test_on_duplicate_attribute(self):
        markup = '<a class="cls" href="url1" href="url2" href="url3" id="id">'
        soup = self.soup(markup)
        assert 'url3' == soup.a['href']
        assert ['cls'] == soup.a['class']
        assert 'id' == soup.a['id']

        def assert_attribute(on_duplicate_attribute, expected):
            soup = self.soup(markup, on_duplicate_attribute=on_duplicate_attribute)
            assert expected == soup.a['href']
            assert ['cls'] == soup.a['class']
            assert 'id' == soup.a['id']
        assert_attribute(None, 'url3')
        assert_attribute(BeautifulSoupHTMLParser.REPLACE, 'url3')
        assert_attribute(BeautifulSoupHTMLParser.IGNORE, 'url1')

        def accumulate(attrs, key, value):
            if not isinstance(attrs[key], list):
                attrs[key] = [attrs[key]]
            attrs[key].append(value)
        assert_attribute(accumulate, ['url1', 'url2', 'url3'])

    def test_html5_attributes(self):
        for input_element, output_unicode, output_element in (('&RightArrowLeftArrow;', '⇄', b'&rlarr;'), ('&models;', '⊧', b'&models;'), ('&Nfr;', '𝔑', b'&Nfr;'), ('&ngeqq;', '≧̸', b'&ngeqq;'), ('&not;', '¬', b'&not;'), ('&Not;', '⫬', b'&Not;'), ('&quot;', '"', b'"'), ('&there4;', '∴', b'&there4;'), ('&Therefore;', '∴', b'&there4;'), ('&therefore;', '∴', b'&there4;'), ('&fjlig;', 'fj', b'fj'), ('&sqcup;', '⊔', b'&sqcup;'), ('&sqcups;', '⊔︀', b'&sqcups;'), ('&apos;', "'", b"'"), ('&verbar;', '|', b'|')):
            markup = '<div>%s</div>' % input_element
            div = self.soup(markup).div
            without_element = div.encode()
            expect = b'<div>%s</div>' % output_unicode.encode('utf8')
            assert without_element == expect
            with_element = div.encode(formatter='html')
            expect = b'<div>%s</div>' % output_element
            assert with_element == expect