import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
@pytest.mark.skipif(not LXML_PRESENT, reason='lxml seems not to be present, not testing its tree builder.')
class TestLXMLTreeBuilder(SoupTest, HTMLTreeBuilderSmokeTest):
    """See ``HTMLTreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        return LXMLTreeBuilder

    def test_out_of_range_entity(self):
        self.assert_soup('<p>foo&#10000000000000;bar</p>', '<p>foobar</p>')
        self.assert_soup('<p>foo&#x10000000000000;bar</p>', '<p>foobar</p>')
        self.assert_soup('<p>foo&#1000000000;bar</p>', '<p>foobar</p>')

    def test_entities_in_foreign_document_encoding(self):
        pass

    @pytest.mark.skipif(not LXML_PRESENT or LXML_VERSION < (2, 3, 5, 0), reason='Skipping doctype test for old version of lxml to avoid segfault.')
    def test_empty_doctype(self):
        soup = self.soup('<!DOCTYPE>')
        doctype = soup.contents[0]
        assert '' == doctype.strip()

    def test_beautifulstonesoup_is_xml_parser(self):
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulStoneSoup('<b />')
        assert '<b/>' == str(soup.b)
        [warning] = w
        assert warning.filename == __file__
        assert 'BeautifulStoneSoup class is deprecated' in str(warning.message)

    def test_tracking_line_numbers(self):
        soup = self.soup('\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>', store_line_numbers=True)
        assert 'sourceline' == soup.p.sourceline.name
        assert 'sourcepos' == soup.p.sourcepos.name