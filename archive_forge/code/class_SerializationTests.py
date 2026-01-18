import re
import sys
import traceback
from collections import OrderedDict
from textwrap import dedent
from types import FunctionType
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, cast
from xml.etree.ElementTree import XML
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.defer import (
from twisted.python.failure import Failure
from twisted.test.testutils import XMLAssertionMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.web._flatten import BUFFER_SIZE
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
class SerializationTests(FlattenTestCase, XMLAssertionMixin):
    """
    Tests for flattening various things.
    """

    def test_nestedTags(self) -> None:
        """
        Test that nested tags flatten correctly.
        """
        self.assertFlattensImmediately(tags.html(tags.body('42'), hi='there'), b'<html hi="there"><body>42</body></html>')

    def test_serializeString(self) -> None:
        """
        Test that strings will be flattened and escaped correctly.
        """
        (self.assertFlattensImmediately('one', b'one'),)
        (self.assertFlattensImmediately('<abc&&>123', b'&lt;abc&amp;&amp;&gt;123'),)

    def test_serializeSelfClosingTags(self) -> None:
        """
        The serialized form of a self-closing tag is C{'<tagName />'}.
        """
        self.assertFlattensImmediately(tags.img(), b'<img />')

    def test_serializeAttribute(self) -> None:
        """
        The serialized form of attribute I{a} with value I{b} is C{'a="b"'}.
        """
        self.assertFlattensImmediately(tags.img(src='foo'), b'<img src="foo" />')

    def test_serializedMultipleAttributes(self) -> None:
        """
        Multiple attributes are separated by a single space in their serialized
        form.
        """
        tag = tags.img()
        tag.attributes = OrderedDict([('src', 'foo'), ('name', 'bar')])
        self.assertFlattensImmediately(tag, b'<img src="foo" name="bar" />')

    def checkAttributeSanitization(self, wrapData: Callable[[str], Flattenable], wrapTag: Callable[[Tag], Flattenable]) -> None:
        """
        Common implementation of L{test_serializedAttributeWithSanitization}
        and L{test_serializedDeferredAttributeWithSanitization},
        L{test_serializedAttributeWithTransparentTag}.

        @param wrapData: A 1-argument callable that wraps around the
            attribute's value so other tests can customize it.

        @param wrapTag: A 1-argument callable that wraps around the outer tag
            so other tests can customize it.
        """
        self.assertFlattensImmediately(wrapTag(tags.img(src=wrapData('<>&"'))), b'<img src="&lt;&gt;&amp;&quot;" />')

    def test_serializedAttributeWithSanitization(self) -> None:
        """
        Attribute values containing C{"<"}, C{">"}, C{"&"}, or C{'"'} have
        C{"&lt;"}, C{"&gt;"}, C{"&amp;"}, or C{"&quot;"} substituted for those
        bytes in the serialized output.
        """
        self.checkAttributeSanitization(passthru, passthru)

    def test_serializedDeferredAttributeWithSanitization(self) -> None:
        """
        Like L{test_serializedAttributeWithSanitization}, but when the contents
        of the attribute are in a L{Deferred
        <twisted.internet.defer.Deferred>}.
        """
        self.checkAttributeSanitization(succeed, passthru)

    def test_serializedAttributeWithSlotWithSanitization(self) -> None:
        """
        Like L{test_serializedAttributeWithSanitization} but with a slot.
        """
        toss = []

        def insertSlot(value: str) -> Flattenable:
            toss.append(value)
            return slot('stuff')

        def fillSlot(tag: Tag) -> Tag:
            return tag.fillSlots(stuff=toss.pop())
        self.checkAttributeSanitization(insertSlot, fillSlot)

    def test_serializedAttributeWithTransparentTag(self) -> None:
        """
        Attribute values which are supplied via the value of a C{t:transparent}
        tag have the same substitution rules to them as values supplied
        directly.
        """
        self.checkAttributeSanitization(tags.transparent, passthru)

    def test_serializedAttributeWithTransparentTagWithRenderer(self) -> None:
        """
        Like L{test_serializedAttributeWithTransparentTag}, but when the
        attribute is rendered by a renderer on an element.
        """

        class WithRenderer(Element):

            def __init__(self, value: str, loader: Optional[ITemplateLoader]) -> None:
                self.value = value
                super().__init__(loader)

            @renderer
            def stuff(self, request: Optional[IRequest], tag: Tag) -> Flattenable:
                return self.value
        toss = []

        def insertRenderer(value: str) -> Flattenable:
            toss.append(value)
            return tags.transparent(render='stuff')

        def render(tag: Tag) -> Flattenable:
            return WithRenderer(toss.pop(), TagLoader(tag))
        self.checkAttributeSanitization(insertRenderer, render)

    def test_serializedAttributeWithRenderable(self) -> None:
        """
        Like L{test_serializedAttributeWithTransparentTag}, but when the
        attribute is a provider of L{IRenderable} rather than a transparent
        tag.
        """

        @implementer(IRenderable)
        class Arbitrary:

            def __init__(self, value: Flattenable) -> None:
                self.value = value

            def render(self, request: Optional[IRequest]) -> Flattenable:
                return self.value

            def lookupRenderMethod(self, name: str) -> Callable[[Optional[IRequest], Tag], Flattenable]:
                raise NotImplementedError('Unexpected call')
        self.checkAttributeSanitization(Arbitrary, passthru)

    def checkTagAttributeSerialization(self, wrapTag: Callable[[Tag], Flattenable]) -> None:
        """
        Common implementation of L{test_serializedAttributeWithTag} and
        L{test_serializedAttributeWithDeferredTag}.

        @param wrapTag: A 1-argument callable that wraps around the attribute's
            value so other tests can customize it.
        @type wrapTag: callable taking L{Tag} and returning something
            flattenable
        """
        innerTag = tags.a('<>&"')
        outerTag = tags.img(src=wrapTag(innerTag))
        outer = self.assertFlattensImmediately(outerTag, b'<img src="&lt;a&gt;&amp;lt;&amp;gt;&amp;amp;&quot;&lt;/a&gt;" />')
        inner = self.assertFlattensImmediately(innerTag, b'<a>&lt;&gt;&amp;"</a>')
        self.assertXMLEqual(XML(outer).attrib['src'], inner)

    def test_serializedAttributeWithTag(self) -> None:
        """
        L{Tag} objects which are serialized within the context of an attribute
        are serialized such that the text content of the attribute may be
        parsed to retrieve the tag.
        """
        self.checkTagAttributeSerialization(passthru)

    def test_serializedAttributeWithDeferredTag(self) -> None:
        """
        Like L{test_serializedAttributeWithTag}, but when the L{Tag} is in a
        L{Deferred <twisted.internet.defer.Deferred>}.
        """
        self.checkTagAttributeSerialization(succeed)

    def test_serializedAttributeWithTagWithAttribute(self) -> None:
        """
        Similar to L{test_serializedAttributeWithTag}, but for the additional
        complexity where the tag which is the attribute value itself has an
        attribute value which contains bytes which require substitution.
        """
        flattened = self.assertFlattensImmediately(tags.img(src=tags.a(href='<>&"')), b'<img src="&lt;a href=&quot;&amp;lt;&amp;gt;&amp;amp;&amp;quot;&quot;&gt;&lt;/a&gt;" />')
        self.assertXMLEqual(XML(flattened).attrib['src'], b'<a href="&lt;&gt;&amp;&quot;"></a>')

    def test_serializeComment(self) -> None:
        """
        Test that comments are correctly flattened and escaped.
        """
        self.assertFlattensImmediately(Comment('foo bar'), b'<!--foo bar-->')

    def test_commentEscaping(self) -> Deferred[List[bytes]]:
        """
        The data in a L{Comment} is escaped and mangled in the flattened output
        so that the result can be safely included in an HTML document.

        Test that C{>} is escaped when the sequence C{-->} is encountered
        within a comment, and that comments do not end with C{-}.
        """

        def verifyComment(c: bytes) -> None:
            self.assertTrue(c.startswith(b'<!--'), f'{c!r} does not start with the comment prefix')
            self.assertTrue(c.endswith(b'-->'), f'{c!r} does not end with the comment suffix')
            self.assertTrue(len(c) >= 7, f'{c!r} is too short to be a legal comment')
            content = c[4:-3]
            if b'foo' in content:
                self.assertIn(b'>', content)
            else:
                self.assertNotIn(b'>', content)
            if content:
                self.assertNotEqual(content[-1], b'-')
        results = []
        for c in ['', 'foo > bar', 'abracadabra-', 'not-->magic']:
            d = flattenString(None, Comment(c))
            d.addCallback(verifyComment)
            results.append(d)
        return gatherResults(results)

    def test_serializeCDATA(self) -> None:
        """
        Test that CDATA is correctly flattened and escaped.
        """
        (self.assertFlattensImmediately(CDATA('foo bar'), b'<![CDATA[foo bar]]>'),)
        self.assertFlattensImmediately(CDATA('foo ]]> bar'), b'<![CDATA[foo ]]]]><![CDATA[> bar]]>')

    def test_serializeUnicode(self) -> None:
        """
        Test that unicode is encoded correctly in the appropriate places, and
        raises an error when it occurs in inappropriate place.
        """
        snowman = '☃'
        self.assertFlattensImmediately(snowman, b'\xe2\x98\x83')
        self.assertFlattensImmediately(tags.p(snowman), b'<p>\xe2\x98\x83</p>')
        self.assertFlattensImmediately(Comment(snowman), b'<!--\xe2\x98\x83-->')
        self.assertFlattensImmediately(CDATA(snowman), b'<![CDATA[\xe2\x98\x83]]>')
        self.assertFlatteningRaises(Tag(snowman), UnicodeEncodeError)
        self.assertFlatteningRaises(Tag('p', attributes={snowman: ''}), UnicodeEncodeError)

    def test_serializeCharRef(self) -> None:
        """
        A character reference is flattened to a string using the I{&#NNNN;}
        syntax.
        """
        ref = CharRef(ord('☃'))
        self.assertFlattensImmediately(ref, b'&#9731;')

    def test_serializeDeferred(self) -> None:
        """
        Test that a deferred is substituted with the current value in the
        callback chain when flattened.
        """
        self.assertFlattensImmediately(succeed('two'), b'two')

    def test_serializeSameDeferredTwice(self) -> None:
        """
        Test that the same deferred can be flattened twice.
        """
        d = succeed('three')
        self.assertFlattensImmediately(d, b'three')
        self.assertFlattensImmediately(d, b'three')

    def test_serializeCoroutine(self) -> None:
        """
        Test that a coroutine returning a value is substituted with the that
        value when flattened.
        """
        from textwrap import dedent
        namespace: Dict[str, FunctionType] = {}
        exec(dedent('\n            async def coro(x):\n                return x\n            '), namespace)
        coro = namespace['coro']
        self.assertFlattensImmediately(coro('four'), b'four')

    def test_serializeCoroutineWithAwait(self) -> None:
        """
        Test that a coroutine returning an awaited deferred value is
        substituted with that value when flattened.
        """
        from textwrap import dedent
        namespace = dict(succeed=succeed)
        exec(dedent('\n            async def coro(x):\n                return await succeed(x)\n            '), namespace)
        coro = namespace['coro']
        self.assertFlattensImmediately(coro('four'), b'four')

    def test_serializeIRenderable(self) -> None:
        """
        Test that flattening respects all of the IRenderable interface.
        """

        @implementer(IRenderable)
        class FakeElement:

            def render(ign, ored: object) -> Tag:
                return tags.p('hello, ', tags.transparent(render='test'), ' - ', tags.transparent(render='test'))

            def lookupRenderMethod(ign, name: str) -> Callable[[Optional[IRequest], Tag], Flattenable]:
                self.assertEqual(name, 'test')
                return lambda ign, node: node('world')
        self.assertFlattensImmediately(FakeElement(), b'<p>hello, world - world</p>')

    def test_serializeMissingRenderFactory(self) -> None:
        """
        Test that flattening a tag with a C{render} attribute when no render
        factory is available in the context raises an exception.
        """
        self.assertFlatteningRaises(tags.transparent(render='test'), ValueError)

    def test_serializeSlots(self) -> None:
        """
        Test that flattening a slot will use the slot value from the tag.
        """
        t1 = tags.p(slot('test'))
        t2 = t1.clone()
        t2.fillSlots(test='hello, world')
        self.assertFlatteningRaises(t1, UnfilledSlot)
        self.assertFlattensImmediately(t2, b'<p>hello, world</p>')

    def test_serializeDeferredSlots(self) -> None:
        """
        Test that a slot with a deferred as its value will be flattened using
        the value from the deferred.
        """
        t = tags.p(slot('test'))
        t.fillSlots(test=succeed(tags.em('four>')))
        self.assertFlattensImmediately(t, b'<p><em>four&gt;</em></p>')

    def test_unknownTypeRaises(self) -> None:
        """
        Test that flattening an unknown type of thing raises an exception.
        """
        self.assertFlatteningRaises(None, UnsupportedType)