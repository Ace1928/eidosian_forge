from typing import Type, Union
from twisted.internet.endpoints import (
from twisted.internet.testing import MemoryReactor
from twisted.trial.unittest import SynchronousTestCase as TestCase
from .._parser import unparseEndpoint
from .._wrapper import HAProxyWrappingFactory
class UnparseEndpointTests(TestCase):
    """
    Tests to ensure that un-parsing an endpoint string round trips through
    escaping properly.
    """

    def check(self, input: str) -> None:
        """
        Check that the input unparses into the output, raising an assertion
        error if it doesn't.

        @param input: an input in endpoint-string-description format.  (To
            ensure determinism, keyword arguments should be in alphabetical
            order.)
        @type input: native L{str}
        """
        self.assertEqual(unparseEndpoint(*parseEndpoint(input)), input)

    def test_basicUnparse(self) -> None:
        """
        An individual word.
        """
        self.check('word')

    def test_multipleArguments(self) -> None:
        """
        Multiple arguments.
        """
        self.check('one:two')

    def test_keywords(self) -> None:
        """
        Keyword arguments.
        """
        self.check('aleph=one:bet=two')

    def test_colonInArgument(self) -> None:
        """
        Escaped ":".
        """
        self.check('hello\\:colon\\:world')

    def test_colonInKeywordValue(self) -> None:
        """
        Escaped ":" in keyword value.
        """
        self.check('hello=\\:')

    def test_colonInKeywordName(self) -> None:
        """
        Escaped ":" in keyword name.
        """
        self.check('\\:=hello')