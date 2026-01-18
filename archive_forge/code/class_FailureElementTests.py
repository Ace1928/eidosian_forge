import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
class FailureElementTests(TestCase):
    """
    Tests for L{FailureElement} and related helpers which can render a
    L{Failure} as an HTML string.
    """

    def setUp(self):
        """
        Create a L{Failure} which can be used by the rendering tests.
        """

        def lineNumberProbeAlsoBroken():
            message = 'This is a problem'
            raise Exception(message)
        self.base = lineNumberProbeAlsoBroken.__code__.co_firstlineno + 1
        try:
            lineNumberProbeAlsoBroken()
        except BaseException:
            self.failure = Failure(captureVars=True)
            self.frame = self.failure.frames[-1]

    def test_sourceLineElement(self):
        """
        L{_SourceLineElement} renders a source line and line number.
        """
        element = _SourceLineElement(TagLoader(tags.div(tags.span(render='lineNumber'), tags.span(render='sourceLine'))), 50, "    print 'hello'")
        d = flattenString(None, element)
        expected = "<div><span>50</span><span> \xa0 \xa0print 'hello'</span></div>"
        d.addCallback(self.assertEqual, expected.encode('utf-8'))
        return d

    def test_sourceFragmentElement(self):
        """
        L{_SourceFragmentElement} renders source lines at and around the line
        number indicated by a frame object.
        """
        element = _SourceFragmentElement(TagLoader(tags.div(tags.span(render='lineNumber'), tags.span(render='sourceLine'), render='sourceLines')), self.frame)
        source = [' \xa0 \xa0message = "This is a problem"', ' \xa0 \xa0raise Exception(message)', '']
        d = flattenString(None, element)
        stringToCheckFor = ''
        for lineNumber, sourceLine in enumerate(source):
            template = '<div class="snippet{}Line"><span>{}</span><span>{}</span></div>'
            if lineNumber <= 1:
                stringToCheckFor += template.format(['', 'Highlight'][lineNumber == 1], self.base + lineNumber, ' \xa0' * 4 + sourceLine)
            else:
                stringToCheckFor += template.format('', self.base + lineNumber, '' + sourceLine)
        bytesToCheckFor = stringToCheckFor.encode('utf8')
        d.addCallback(self.assertEqual, bytesToCheckFor)
        return d

    def test_frameElementFilename(self):
        """
        The I{filename} renderer of L{_FrameElement} renders the filename
        associated with the frame object used to initialize the
        L{_FrameElement}.
        """
        element = _FrameElement(TagLoader(tags.span(render='filename')), self.frame)
        d = flattenString(None, element)
        d.addCallback(self.assertEqual, b'<span>' + networkString(__file__.rstrip('c')) + b'</span>')
        return d

    def test_frameElementLineNumber(self):
        """
        The I{lineNumber} renderer of L{_FrameElement} renders the line number
        associated with the frame object used to initialize the
        L{_FrameElement}.
        """
        element = _FrameElement(TagLoader(tags.span(render='lineNumber')), self.frame)
        d = flattenString(None, element)
        d.addCallback(self.assertEqual, b'<span>%d</span>' % (self.base + 1,))
        return d

    def test_frameElementFunction(self):
        """
        The I{function} renderer of L{_FrameElement} renders the line number
        associated with the frame object used to initialize the
        L{_FrameElement}.
        """
        element = _FrameElement(TagLoader(tags.span(render='function')), self.frame)
        d = flattenString(None, element)
        d.addCallback(self.assertEqual, b'<span>lineNumberProbeAlsoBroken</span>')
        return d

    def test_frameElementSource(self):
        """
        The I{source} renderer of L{_FrameElement} renders the source code near
        the source filename/line number associated with the frame object used to
        initialize the L{_FrameElement}.
        """
        element = _FrameElement(None, self.frame)
        renderer = element.lookupRenderMethod('source')
        tag = tags.div()
        result = renderer(None, tag)
        self.assertIsInstance(result, _SourceFragmentElement)
        self.assertIdentical(result.frame, self.frame)
        self.assertEqual([tag], result.loader.load())

    def test_stackElement(self):
        """
        The I{frames} renderer of L{_StackElement} renders each stack frame in
        the list of frames used to initialize the L{_StackElement}.
        """
        element = _StackElement(None, self.failure.frames[:2])
        renderer = element.lookupRenderMethod('frames')
        tag = tags.div()
        result = renderer(None, tag)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], _FrameElement)
        self.assertIdentical(result[0].frame, self.failure.frames[0])
        self.assertIsInstance(result[1], _FrameElement)
        self.assertIdentical(result[1].frame, self.failure.frames[1])
        self.assertNotEqual(result[0].loader.load(), result[1].loader.load())
        self.assertEqual(2, len(result))

    def test_failureElementTraceback(self):
        """
        The I{traceback} renderer of L{FailureElement} renders the failure's
        stack frames using L{_StackElement}.
        """
        element = FailureElement(self.failure)
        renderer = element.lookupRenderMethod('traceback')
        tag = tags.div()
        result = renderer(None, tag)
        self.assertIsInstance(result, _StackElement)
        self.assertIdentical(result.stackFrames, self.failure.frames)
        self.assertEqual([tag], result.loader.load())

    def test_failureElementType(self):
        """
        The I{type} renderer of L{FailureElement} renders the failure's
        exception type.
        """
        element = FailureElement(self.failure, TagLoader(tags.span(render='type')))
        d = flattenString(None, element)
        exc = b'builtins.Exception'
        d.addCallback(self.assertEqual, b'<span>' + exc + b'</span>')
        return d

    def test_failureElementValue(self):
        """
        The I{value} renderer of L{FailureElement} renders the value's exception
        value.
        """
        element = FailureElement(self.failure, TagLoader(tags.span(render='value')))
        d = flattenString(None, element)
        d.addCallback(self.assertEqual, b'<span>This is a problem</span>')
        return d