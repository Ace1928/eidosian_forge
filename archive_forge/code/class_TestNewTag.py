from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
class TestNewTag(SoupTest):
    """Test the BeautifulSoup.new_tag() method."""

    def test_new_tag(self):
        soup = self.soup('')
        new_tag = soup.new_tag('foo', bar='baz', attrs={'name': 'a name'})
        assert isinstance(new_tag, Tag)
        assert 'foo' == new_tag.name
        assert dict(bar='baz', name='a name') == new_tag.attrs
        assert None == new_tag.parent

    @pytest.mark.skipif(not LXML_PRESENT, reason='lxml not installed, cannot parse XML document')
    def test_xml_tag_inherits_self_closing_rules_from_builder(self):
        xml_soup = BeautifulSoup('', 'xml')
        xml_br = xml_soup.new_tag('br')
        xml_p = xml_soup.new_tag('p')
        assert b'<br/>' == xml_br.encode()
        assert b'<p/>' == xml_p.encode()

    def test_tag_inherits_self_closing_rules_from_builder(self):
        html_soup = BeautifulSoup('', 'html.parser')
        html_br = html_soup.new_tag('br')
        html_p = html_soup.new_tag('p')
        assert b'<br/>' == html_br.encode()
        assert b'<p></p>' == html_p.encode()