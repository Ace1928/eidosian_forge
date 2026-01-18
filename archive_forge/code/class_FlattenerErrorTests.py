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
class FlattenerErrorTests(SynchronousTestCase):
    """
    Tests for L{FlattenerError}.
    """

    def test_renderable(self) -> None:
        """
        If a L{FlattenerError} is created with an L{IRenderable} provider root,
        the repr of that object is included in the string representation of the
        exception.
        """

        @implementer(IRenderable)
        class Renderable:

            def __repr__(self) -> str:
                return 'renderable repr'
        self.assertEqual(str(FlattenerError(RuntimeError('reason'), [Renderable()], [])), 'Exception while flattening:\n  renderable repr\nRuntimeError: reason\n')

    def test_tag(self) -> None:
        """
        If a L{FlattenerError} is created with a L{Tag} instance with source
        location information, the source location is included in the string
        representation of the exception.
        """
        tag = Tag('div', filename='/foo/filename.xhtml', lineNumber=17, columnNumber=12)
        self.assertEqual(str(FlattenerError(RuntimeError('reason'), [tag], [])), 'Exception while flattening:\n  File "/foo/filename.xhtml", line 17, column 12, in "div"\nRuntimeError: reason\n')

    def test_tagWithoutLocation(self) -> None:
        """
        If a L{FlattenerError} is created with a L{Tag} instance without source
        location information, only the tagName is included in the string
        representation of the exception.
        """
        self.assertEqual(str(FlattenerError(RuntimeError('reason'), [Tag('span')], [])), 'Exception while flattening:\n  Tag <span>\nRuntimeError: reason\n')

    def test_traceback(self) -> None:
        """
        If a L{FlattenerError} is created with traceback frames, they are
        included in the string representation of the exception.
        """

        def f() -> None:
            g()

        def g() -> NoReturn:
            raise RuntimeError('reason')
        try:
            f()
        except RuntimeError as e:
            tbinfo = traceback.extract_tb(sys.exc_info()[2])[1:]
            exc = e
        else:
            self.fail('f() must raise RuntimeError')
        self.assertEqual(str(FlattenerError(exc, [], tbinfo)), 'Exception while flattening:\n  File "%s", line %d, in f\n    g()\n  File "%s", line %d, in g\n    raise RuntimeError("reason")\nRuntimeError: reason\n' % (HERE, f.__code__.co_firstlineno + 1, HERE, g.__code__.co_firstlineno + 1))

    def test_asynchronousFlattenError(self) -> None:
        """
        When flattening a renderer which raises an exception asynchronously,
        the error is reported when it occurs.
        """
        failing: Deferred[object] = Deferred()

        @implementer(IRenderable)
        class NotActuallyRenderable:
            """No methods provided; this will fail"""

            def __repr__(self) -> str:
                return '<unrenderable>'

            def lookupRenderMethod(self, name: str) -> Callable[[Optional[IRequest], Tag], Flattenable]:
                ...

            def render(self, request: Optional[IRequest]) -> Flattenable:
                return failing
        flattening = flattenString(None, [NotActuallyRenderable()])
        self.assertNoResult(flattening)
        exc = RuntimeError('example')
        failing.errback(exc)
        failure = self.failureResultOf(flattening, FlattenerError)
        self.assertRegex(str(failure.value), re.compile(dedent('                    Exception while flattening:\n                      \\[<unrenderable>\\]\n                      <unrenderable>\n                      <Deferred at .* current result: <twisted.python.failure.Failure builtins.RuntimeError: example>>\n                      File ".*", line \\d*, in _flattenTree\n                        element = await element.*\n                    '), flags=re.MULTILINE))
        self.assertIn('RuntimeError: example', str(failure.value))
        self.failureResultOf(failing, RuntimeError)

    def test_cancel(self) -> None:
        """
        The flattening of a Deferred can be cancelled.
        """
        cancelCount = 0
        cancelArg = None

        def checkCancel(cancelled: Deferred[object]) -> None:
            nonlocal cancelArg, cancelCount
            cancelArg = cancelled
            cancelCount += 1
        err = None

        def saveErr(failure: Failure) -> None:
            nonlocal err
            err = failure
        d: Deferred[object] = Deferred(checkCancel)
        flattening = flattenString(None, d)
        self.assertNoResult(flattening)
        d.addErrback(saveErr)
        flattening.cancel()
        failure = self.failureResultOf(flattening, FlattenerError)
        self.assertEqual(cancelCount, 1)
        self.assertIs(cancelArg, d)
        self.assertIsInstance(err, Failure)
        self.assertIsInstance(cast(Failure, err).value, CancelledError)
        exc = failure.value.args[0]
        self.assertIsInstance(exc, CancelledError)