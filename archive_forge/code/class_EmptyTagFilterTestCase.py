import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
class EmptyTagFilterTestCase(unittest.TestCase):

    def test_empty(self):
        stream = XML('<elem></elem>') | EmptyTagFilter()
        self.assertEqual([EmptyTagFilter.EMPTY], [ev[0] for ev in stream])

    def test_text_content(self):
        stream = XML('<elem>foo</elem>') | EmptyTagFilter()
        self.assertEqual([Stream.START, Stream.TEXT, Stream.END], [ev[0] for ev in stream])

    def test_elem_content(self):
        stream = XML('<elem><sub /><sub /></elem>') | EmptyTagFilter()
        self.assertEqual([Stream.START, EmptyTagFilter.EMPTY, EmptyTagFilter.EMPTY, Stream.END], [ev[0] for ev in stream])