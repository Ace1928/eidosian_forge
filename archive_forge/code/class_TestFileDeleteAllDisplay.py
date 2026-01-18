from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestFileDeleteAllDisplay(TestCase):

    def test_filedeleteall(self):
        c = commands.FileDeleteAllCommand()
        self.assertEqual(b'deleteall', bytes(c))