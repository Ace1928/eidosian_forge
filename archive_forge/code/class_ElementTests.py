from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
class ElementTests(unittest.TestCase):
    """
    Tests for L{domish.Element}.
    """

    def test_interface(self):
        """
        L{domish.Element} implements L{domish.IElement}.
        """
        verifyObject(domish.IElement, domish.Element((None, 'foo')))

    def test_escaping(self):
        """
        The built-in entity references are properly encoded.
        """
        s = '&<>\'"'
        self.assertEqual(domish.escapeToXml(s), '&amp;&lt;&gt;\'"')
        self.assertEqual(domish.escapeToXml(s, 1), '&amp;&lt;&gt;&apos;&quot;')

    def test_namespace(self):
        """
        An attribute on L{domish.Namespace} yields a qualified name.
        """
        ns = domish.Namespace('testns')
        self.assertEqual(ns.foo, ('testns', 'foo'))

    def test_elementInit(self):
        """
        Basic L{domish.Element} initialization tests.
        """
        e = domish.Element((None, 'foo'))
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, None)
        self.assertEqual(e.defaultUri, None)
        self.assertEqual(e.parent, None)
        e = domish.Element(('', 'foo'))
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, '')
        self.assertEqual(e.defaultUri, '')
        self.assertEqual(e.parent, None)
        e = domish.Element(('testns', 'foo'))
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, 'testns')
        self.assertEqual(e.defaultUri, 'testns')
        self.assertEqual(e.parent, None)
        e = domish.Element(('testns', 'foo'), 'test2ns')
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, 'testns')
        self.assertEqual(e.defaultUri, 'test2ns')

    def test_childOps(self):
        """
        Basic L{domish.Element} child tests.
        """
        e = domish.Element(('testns', 'foo'))
        e.addContent('somecontent')
        b2 = e.addElement(('testns2', 'bar2'))
        e['attrib1'] = 'value1'
        e['testns2', 'attrib2'] = 'value2'
        e.addElement('bar')
        e.addElement('bar')
        e.addContent('abc')
        e.addContent('123')
        self.assertEqual(e.children[-1], 'abc123')
        self.assertEqual(e.bar2, b2)
        e.bar2.addContent('subcontent')
        e.bar2['bar2value'] = 'somevalue'
        self.assertEqual(e.children[1], e.bar2)
        self.assertEqual(e.children[2], e.bar)
        self.assertEqual(e['attrib1'], 'value1')
        del e['attrib1']
        self.assertEqual(e.hasAttribute('attrib1'), 0)
        self.assertEqual(e.hasAttribute('attrib2'), 0)
        self.assertEqual(e['testns2', 'attrib2'], 'value2')

    def test_characterData(self):
        """
        Extract character data using L{str}.
        """
        element = domish.Element(('testns', 'foo'))
        element.addContent('somecontent')
        text = str(element)
        self.assertEqual('somecontent', text)
        self.assertIsInstance(text, str)

    def test_characterDataNativeString(self):
        """
        Extract ascii character data using L{str}.
        """
        element = domish.Element(('testns', 'foo'))
        element.addContent('somecontent')
        text = str(element)
        self.assertEqual('somecontent', text)
        self.assertIsInstance(text, str)

    def test_characterDataUnicode(self):
        """
        Extract character data using L{str}.
        """
        element = domish.Element(('testns', 'foo'))
        element.addContent('☃')
        text = str(element)
        self.assertEqual('☃', text)
        self.assertIsInstance(text, str)

    def test_characterDataBytes(self):
        """
        Extract character data as UTF-8 using L{bytes}.
        """
        element = domish.Element(('testns', 'foo'))
        element.addContent('☃')
        text = bytes(element)
        self.assertEqual('☃'.encode(), text)
        self.assertIsInstance(text, bytes)

    def test_characterDataMixed(self):
        """
        Mixing addChild with cdata and element, the first cdata is returned.
        """
        element = domish.Element(('testns', 'foo'))
        element.addChild('abc')
        element.addElement('bar')
        element.addChild('def')
        self.assertEqual('abc', str(element))

    def test_addContent(self):
        """
        Unicode strings passed to C{addContent} become the character data.
        """
        element = domish.Element(('testns', 'foo'))
        element.addContent('unicode')
        self.assertEqual('unicode', str(element))

    def test_addContentNativeStringASCII(self):
        """
        ASCII native strings passed to C{addContent} become the character data.
        """
        element = domish.Element(('testns', 'foo'))
        element.addContent('native')
        self.assertEqual('native', str(element))

    def test_addContentBytes(self):
        """
        Byte strings passed to C{addContent} are not acceptable on Python 3.
        """
        element = domish.Element(('testns', 'foo'))
        self.assertRaises(TypeError, element.addContent, b'bytes')

    def test_addElementContent(self):
        """
        Content passed to addElement becomes character data on the new child.
        """
        element = domish.Element(('testns', 'foo'))
        child = element.addElement('bar', content='abc')
        self.assertEqual('abc', str(child))

    def test_elements(self):
        """
        Calling C{elements} without arguments on a L{domish.Element} returns
        all child elements, whatever the qualified name.
        """
        e = domish.Element(('testns', 'foo'))
        c1 = e.addElement('name')
        c2 = e.addElement(('testns2', 'baz'))
        c3 = e.addElement('quux')
        c4 = e.addElement(('testns', 'name'))
        elts = list(e.elements())
        self.assertIn(c1, elts)
        self.assertIn(c2, elts)
        self.assertIn(c3, elts)
        self.assertIn(c4, elts)

    def test_elementsWithQN(self):
        """
        Calling C{elements} with a namespace and local name on a
        L{domish.Element} returns all child elements with that qualified name.
        """
        e = domish.Element(('testns', 'foo'))
        c1 = e.addElement('name')
        c2 = e.addElement(('testns2', 'baz'))
        c3 = e.addElement('quux')
        c4 = e.addElement(('testns', 'name'))
        elts = list(e.elements('testns', 'name'))
        self.assertIn(c1, elts)
        self.assertNotIn(c2, elts)
        self.assertNotIn(c3, elts)
        self.assertIn(c4, elts)