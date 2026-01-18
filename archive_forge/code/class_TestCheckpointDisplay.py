from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestCheckpointDisplay(TestCase):

    def test_checkpoint(self):
        c = commands.CheckpointCommand()
        self.assertEqual(b'checkpoint', bytes(c))