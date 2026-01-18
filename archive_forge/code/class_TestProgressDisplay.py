from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestProgressDisplay(TestCase):

    def test_progress(self):
        c = commands.ProgressCommand(b'doing foo')
        self.assertEqual(b'progress doing foo', bytes(c))