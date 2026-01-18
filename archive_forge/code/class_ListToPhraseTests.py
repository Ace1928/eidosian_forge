from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
class ListToPhraseTests(SynchronousTestCase):
    """
    Input is transformed into a string representation of the list,
    with each item separated by delimiter (defaulting to a comma) and the final
    two being separated by a final delimiter.
    """

    def test_empty(self) -> None:
        """
        If things is empty, an empty string is returned.
        """
        sample: list[None] = []
        expected = ''
        result = util._listToPhrase(sample, 'and')
        self.assertEqual(expected, result)

    def test_oneWord(self) -> None:
        """
        With a single item, the item is returned.
        """
        sample = ['One']
        expected = 'One'
        result = util._listToPhrase(sample, 'and')
        self.assertEqual(expected, result)

    def test_twoWords(self) -> None:
        """
        Two words are separated by the final delimiter.
        """
        sample = ['One', 'Two']
        expected = 'One and Two'
        result = util._listToPhrase(sample, 'and')
        self.assertEqual(expected, result)

    def test_threeWords(self) -> None:
        """
        With more than two words, the first two are separated by the delimiter.
        """
        sample = ['One', 'Two', 'Three']
        expected = 'One, Two, and Three'
        result = util._listToPhrase(sample, 'and')
        self.assertEqual(expected, result)

    def test_fourWords(self) -> None:
        """
        If a delimiter is specified, it is used instead of the default comma.
        """
        sample = ['One', 'Two', 'Three', 'Four']
        expected = 'One; Two; Three; or Four'
        result = util._listToPhrase(sample, 'or', delimiter='; ')
        self.assertEqual(expected, result)

    def test_notString(self) -> None:
        """
        If something in things is not a string, it is converted into one.
        """
        sample = [1, 2, 'three']
        expected = '1, 2, and three'
        result = util._listToPhrase(sample, 'and')
        self.assertEqual(expected, result)

    def test_stringTypeError(self) -> None:
        """
        If things is a string, a TypeError is raised.
        """
        sample = 'One, two, three'
        error = self.assertRaises(TypeError, util._listToPhrase, sample, 'and')
        self.assertEqual(str(error), 'Things must be a list or a tuple')

    def test_iteratorTypeError(self) -> None:
        """
        If things is an iterator, a TypeError is raised.
        """
        sample = iter([1, 2, 3])
        error = self.assertRaises(TypeError, util._listToPhrase, sample, 'and')
        self.assertEqual(str(error), 'Things must be a list or a tuple')

    def test_generatorTypeError(self) -> None:
        """
        If things is a generator, a TypeError is raised.
        """

        def sample() -> Generator[int, None, None]:
            yield from range(2)
        error = self.assertRaises(TypeError, util._listToPhrase, sample, 'and')
        self.assertEqual(str(error), 'Things must be a list or a tuple')