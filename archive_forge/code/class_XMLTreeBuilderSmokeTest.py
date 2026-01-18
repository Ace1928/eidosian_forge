import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
class XMLTreeBuilderSmokeTest(TreeBuilderSmokeTest):

    def test_pickle_and_unpickle_identity(self):
        tree = self.soup('<a><b>foo</a>')
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == tree.decode()

    def test_docstring_generated(self):
        soup = self.soup('<root/>')
        assert soup.encode() == b'<?xml version="1.0" encoding="utf-8"?>\n<root/>'

    def test_xml_declaration(self):
        markup = b'<?xml version="1.0" encoding="utf8"?>\n<foo/>'
        soup = self.soup(markup)
        assert markup == soup.encode('utf8')

    def test_python_specific_encodings_not_used_in_xml_declaration(self):
        markup = b'<?xml version="1.0"?>\n<foo/>'
        soup = self.soup(markup)
        for encoding in PYTHON_SPECIFIC_ENCODINGS:
            if encoding in ('idna', 'mbcs', 'oem', 'undefined', 'string_escape', 'string-escape'):
                continue
            encoded = soup.encode(encoding)
            assert b'<?xml version="1.0"?>' in encoded
            assert encoding.encode('ascii') not in encoded

    def test_processing_instruction(self):
        markup = b'<?xml version="1.0" encoding="utf8"?>\n<?PITarget PIContent?>'
        soup = self.soup(markup)
        assert markup == soup.encode('utf8')

    def test_real_xhtml_document(self):
        """A real XHTML document should come out *exactly* the same as it went in."""
        markup = b'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">\n<html xmlns="http://www.w3.org/1999/xhtml">\n<head><title>Hello.</title></head>\n<body>Goodbye.</body>\n</html>'
        soup = self.soup(markup)
        assert soup.encode('utf-8') == markup

    def test_nested_namespaces(self):
        doc = b'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n<parent xmlns="http://ns1/">\n<child xmlns="http://ns2/" xmlns:ns3="http://ns3/">\n<grandchild ns3:attr="value" xmlns="http://ns4/"/>\n</child>\n</parent>'
        soup = self.soup(doc)
        assert doc == soup.encode()

    def test_formatter_processes_script_tag_for_xml_documents(self):
        doc = '\n  <script type="text/javascript">\n  </script>\n'
        soup = BeautifulSoup(doc, 'lxml-xml')
        soup.script.string = 'console.log("< < hey > > ");'
        encoded = soup.encode()
        assert b'&lt; &lt; hey &gt; &gt;' in encoded

    def test_can_parse_unicode_document(self):
        markup = '<?xml version="1.0" encoding="euc-jp"><root>Sacré bleu!</root>'
        soup = self.soup(markup)
        assert 'Sacré bleu!' == soup.root.string

    def test_can_parse_unicode_document_begining_with_bom(self):
        markup = '\ufeff<?xml version="1.0" encoding="euc-jp"><root>Sacré bleu!</root>'
        soup = self.soup(markup)
        assert 'Sacré bleu!' == soup.root.string

    def test_popping_namespaced_tag(self):
        markup = '<rss xmlns:dc="foo"><dc:creator>b</dc:creator><dc:date>2012-07-02T20:33:42Z</dc:date><dc:rights>c</dc:rights><image>d</image></rss>'
        soup = self.soup(markup)
        assert str(soup.rss) == markup

    def test_docstring_includes_correct_encoding(self):
        soup = self.soup('<root/>')
        assert soup.encode('latin1') == b'<?xml version="1.0" encoding="latin1"?>\n<root/>'

    def test_large_xml_document(self):
        """A large XML document should come out the same as it went in."""
        markup = b'<?xml version="1.0" encoding="utf-8"?>\n<root>' + b'0' * 2 ** 12 + b'</root>'
        soup = self.soup(markup)
        assert soup.encode('utf-8') == markup

    def test_tags_are_empty_element_if_and_only_if_they_are_empty(self):
        self.assert_soup('<p>', '<p/>')
        self.assert_soup('<p>foo</p>')

    def test_namespaces_are_preserved(self):
        markup = '<root xmlns:a="http://example.com/" xmlns:b="http://example.net/"><a:foo>This tag is in the a namespace</a:foo><b:foo>This tag is in the b namespace</b:foo></root>'
        soup = self.soup(markup)
        root = soup.root
        assert 'http://example.com/' == root['xmlns:a']
        assert 'http://example.net/' == root['xmlns:b']

    def test_closing_namespaced_tag(self):
        markup = '<p xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:date>20010504</dc:date></p>'
        soup = self.soup(markup)
        assert str(soup.p) == markup

    def test_namespaced_attributes(self):
        markup = '<foo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><bar xsi:schemaLocation="http://www.example.com"/></foo>'
        soup = self.soup(markup)
        assert str(soup.foo) == markup

    def test_namespaced_attributes_xml_namespace(self):
        markup = '<foo xml:lang="fr">bar</foo>'
        soup = self.soup(markup)
        assert str(soup.foo) == markup

    def test_find_by_prefixed_name(self):
        doc = '<?xml version="1.0" encoding="utf-8"?>\n<Document xmlns="http://example.com/ns0"\n    xmlns:ns1="http://example.com/ns1"\n    xmlns:ns2="http://example.com/ns2">\n    <ns1:tag>foo</ns1:tag>\n    <ns1:tag>bar</ns1:tag>\n    <ns2:tag key="value">baz</ns2:tag>\n</Document>\n'
        soup = self.soup(doc)
        assert 3 == len(soup.find_all('tag'))
        assert 2 == len(soup.find_all('ns1:tag'))
        assert 1 == len(soup.find_all('ns2:tag'))
        assert 1, len(soup.find_all('ns2:tag', key='value'))
        assert 3, len(soup.find_all(['ns1:tag', 'ns2:tag']))

    def test_copy_tag_preserves_namespace(self):
        xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document xmlns:w="http://example.com/ns0"/>'
        soup = self.soup(xml)
        tag = soup.document
        duplicate = copy.copy(tag)
        assert tag.prefix == duplicate.prefix

    def test_worst_case(self):
        """Test the worst case (currently) for linking issues."""
        soup = self.soup(BAD_DOCUMENT)
        self.linkage_validator(soup)