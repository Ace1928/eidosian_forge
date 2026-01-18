import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION
from bs4 import (
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
@pytest.mark.skipif(not LXML_PRESENT, reason='lxml seems not to be present, not testing its XML tree builder.')
class TestLXMLXMLTreeBuilder(SoupTest, XMLTreeBuilderSmokeTest):
    """See ``HTMLTreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        return LXMLTreeBuilderForXML

    def test_namespace_indexing(self):
        soup = self.soup('<?xml version="1.1"?>\n<root><tag xmlns="http://unprefixed-namespace.com">content</tag><prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</prefix:tag2><prefix2:tag3 xmlns:prefix2="http://another-namespace.com"><subtag xmlns="http://another-unprefixed-namespace.com"><subsubtag xmlns="http://yet-another-unprefixed-namespace.com"></prefix2:tag3></root>')
        assert soup._namespaces == {'xml': 'http://www.w3.org/XML/1998/namespace', 'prefix': 'http://prefixed-namespace.com', 'prefix2': 'http://another-namespace.com'}
        assert soup.tag._namespaces == {'xml': 'http://www.w3.org/XML/1998/namespace'}
        assert soup.tag2._namespaces == {'prefix': 'http://prefixed-namespace.com', 'xml': 'http://www.w3.org/XML/1998/namespace'}
        assert soup.subtag._namespaces == {'prefix2': 'http://another-namespace.com', 'xml': 'http://www.w3.org/XML/1998/namespace'}
        assert soup.subsubtag._namespaces == {'prefix2': 'http://another-namespace.com', 'xml': 'http://www.w3.org/XML/1998/namespace'}

    @pytest.mark.skipif(not SOUP_SIEVE_PRESENT, reason='Soup Sieve not installed')
    def test_namespace_interaction_with_select_and_find(self):
        soup = self.soup('<?xml version="1.1"?>\n<root><tag xmlns="http://unprefixed-namespace.com">content</tag><prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</tag><subtag xmlns:prefix="http://another-namespace-same-prefix.com"><prefix:tag3></subtag></root>')
        assert soup.select_one('tag').name == 'tag'
        assert soup.select_one('prefix|tag2').name == 'tag2'
        assert soup.select_one('prefix|tag3') is None
        assert soup.select_one('prefix|tag3', namespaces=soup.subtag._namespaces).name == 'tag3'
        assert soup.subtag.select_one('prefix|tag3').name == 'tag3'
        assert soup.find('tag').name == 'tag'
        assert soup.find('prefix:tag2').name == 'tag2'
        assert soup.find('prefix:tag3').name == 'tag3'
        assert soup.subtag.find('prefix:tag3').name == 'tag3'

    def test_pickle_restores_builder(self):
        soup = self.soup('<a>some markup</a>')
        assert isinstance(soup.builder, self.default_builder)
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert 'some markup' == unpickled.a.string
        assert unpickled.builder != soup.builder
        assert isinstance(unpickled.builder, self.default_builder)