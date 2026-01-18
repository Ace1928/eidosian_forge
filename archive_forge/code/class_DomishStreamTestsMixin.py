from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
class DomishStreamTestsMixin:
    """
    Mixin defining tests for different stream implementations.

    @ivar streamClass: A no-argument callable which will be used to create an
        XML parser which can produce a stream of elements from incremental
        input.
    """

    def setUp(self):
        self.doc_started = False
        self.doc_ended = False
        self.root = None
        self.elements = []
        self.stream = self.streamClass()
        self.stream.DocumentStartEvent = self._docStarted
        self.stream.ElementEvent = self.elements.append
        self.stream.DocumentEndEvent = self._docEnded

    def _docStarted(self, root):
        self.root = root
        self.doc_started = True

    def _docEnded(self):
        self.doc_ended = True

    def doTest(self, xml):
        self.stream.parse(xml)

    def testHarness(self):
        xml = b'<root><child/><child2/></root>'
        self.stream.parse(xml)
        self.assertEqual(self.doc_started, True)
        self.assertEqual(self.root.name, 'root')
        self.assertEqual(self.elements[0].name, 'child')
        self.assertEqual(self.elements[1].name, 'child2')
        self.assertEqual(self.doc_ended, True)

    def testBasic(self):
        xml = b"<stream:stream xmlns:stream='etherx' xmlns='jabber'>\n" + b"  <message to='bar'>" + b"    <x xmlns='xdelay'>some&amp;data&gt;</x>" + b'  </message>' + b'</stream:stream>'
        self.stream.parse(xml)
        self.assertEqual(self.root.name, 'stream')
        self.assertEqual(self.root.uri, 'etherx')
        self.assertEqual(self.elements[0].name, 'message')
        self.assertEqual(self.elements[0].uri, 'jabber')
        self.assertEqual(self.elements[0]['to'], 'bar')
        self.assertEqual(self.elements[0].x.uri, 'xdelay')
        self.assertEqual(str(self.elements[0].x), 'some&data>')

    def testNoRootNS(self):
        xml = b"<stream><error xmlns='etherx'/></stream>"
        self.stream.parse(xml)
        self.assertEqual(self.root.uri, '')
        self.assertEqual(self.elements[0].uri, 'etherx')

    def testNoDefaultNS(self):
        xml = b"<stream:stream xmlns:stream='etherx'><error/></stream:stream>"
        self.stream.parse(xml)
        self.assertEqual(self.root.uri, 'etherx')
        self.assertEqual(self.root.defaultUri, '')
        self.assertEqual(self.elements[0].uri, '')
        self.assertEqual(self.elements[0].defaultUri, '')

    def testChildDefaultNS(self):
        xml = b"<root xmlns='testns'><child/></root>"
        self.stream.parse(xml)
        self.assertEqual(self.root.uri, 'testns')
        self.assertEqual(self.elements[0].uri, 'testns')

    def testEmptyChildNS(self):
        xml = b"<root xmlns='testns'>\n                    <child1><child2 xmlns=''/></child1>\n                  </root>"
        self.stream.parse(xml)
        self.assertEqual(self.elements[0].child2.uri, '')

    def test_attributesWithNamespaces(self):
        """
        Attributes with namespace are parsed without Exception.
        (https://twistedmatrix.com/trac/ticket/9730 regression test)
        """
        xml = b"<root xmlns:test='http://example.org' xml:lang='en'>\n                    <test:test>test</test:test>\n                  </root>"
        self.stream.parse(xml)
        self.assertEqual(self.elements[0].uri, 'http://example.org')

    def testChildPrefix(self):
        xml = b"<root xmlns='testns' xmlns:foo='testns2'><foo:child/></root>"
        self.stream.parse(xml)
        self.assertEqual(self.root.localPrefixes['foo'], 'testns2')
        self.assertEqual(self.elements[0].uri, 'testns2')

    def testUnclosedElement(self):
        self.assertRaises(domish.ParserError, self.stream.parse, b'<root><error></root>')

    def test_namespaceReuse(self):
        """
        Test that reuse of namespaces does affect an element's serialization.

        When one element uses a prefix for a certain namespace, this is
        stored in the C{localPrefixes} attribute of the element. We want
        to make sure that elements created after such use, won't have this
        prefix end up in their C{localPrefixes} attribute, too.
        """
        xml = b"<root>\n                    <foo:child1 xmlns:foo='testns'/>\n                    <child2 xmlns='testns'/>\n                  </root>"
        self.stream.parse(xml)
        self.assertEqual('child1', self.elements[0].name)
        self.assertEqual('testns', self.elements[0].uri)
        self.assertEqual('', self.elements[0].defaultUri)
        self.assertEqual({'foo': 'testns'}, self.elements[0].localPrefixes)
        self.assertEqual('child2', self.elements[1].name)
        self.assertEqual('testns', self.elements[1].uri)
        self.assertEqual('testns', self.elements[1].defaultUri)
        self.assertEqual({}, self.elements[1].localPrefixes)