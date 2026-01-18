import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
class TestMatchesIterator(unittest.TestCase):

    def setUp(self):
        self.matches = ['bobby', 'bobbies', 'bobberina']
        self.matches_iterator = repl.MatchesIterator()
        self.matches_iterator.current_word = 'bob'
        self.matches_iterator.orig_line = 'bob'
        self.matches_iterator.orig_cursor_offset = len('bob')
        self.matches_iterator.matches = self.matches

    def test_next(self):
        self.assertEqual(next(self.matches_iterator), self.matches[0])
        for x in range(len(self.matches) - 1):
            next(self.matches_iterator)
        self.assertEqual(next(self.matches_iterator), self.matches[0])
        self.assertEqual(next(self.matches_iterator), self.matches[1])
        self.assertNotEqual(next(self.matches_iterator), self.matches[1])

    def test_previous(self):
        self.assertEqual(self.matches_iterator.previous(), self.matches[2])
        for x in range(len(self.matches) - 1):
            self.matches_iterator.previous()
        self.assertNotEqual(self.matches_iterator.previous(), self.matches[0])
        self.assertEqual(self.matches_iterator.previous(), self.matches[1])
        self.assertEqual(self.matches_iterator.previous(), self.matches[0])

    def test_nonzero(self):
        """self.matches_iterator should be False at start,
        then True once we active a match.
        """
        self.assertFalse(self.matches_iterator)
        next(self.matches_iterator)
        self.assertTrue(self.matches_iterator)

    def test_iter(self):
        slice = islice(self.matches_iterator, 0, 9)
        self.assertEqual(list(slice), self.matches * 3)

    def test_current(self):
        with self.assertRaises(ValueError):
            self.matches_iterator.current()
        next(self.matches_iterator)
        self.assertEqual(self.matches_iterator.current(), self.matches[0])

    def test_update(self):
        slice = islice(self.matches_iterator, 0, 3)
        self.assertEqual(list(slice), self.matches)
        newmatches = ['string', 'str', 'set']
        completer = mock.Mock()
        completer.locate.return_value = LinePart(0, 1, 's')
        self.matches_iterator.update(1, 's', newmatches, completer)
        newslice = islice(newmatches, 0, 3)
        self.assertNotEqual(list(slice), self.matches)
        self.assertEqual(list(newslice), newmatches)

    def test_cur_line(self):
        completer = mock.Mock()
        completer.locate.return_value = LinePart(0, self.matches_iterator.orig_cursor_offset, self.matches_iterator.orig_line)
        self.matches_iterator.completer = completer
        with self.assertRaises(ValueError):
            self.matches_iterator.cur_line()
        self.assertEqual(next(self.matches_iterator), self.matches[0])
        self.assertEqual(self.matches_iterator.cur_line(), (len(self.matches[0]), self.matches[0]))

    def test_is_cseq(self):
        self.assertTrue(self.matches_iterator.is_cseq())