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
class TestEditConfig(TestCase):

    def setUp(self):
        self.repl = FakeRepl()
        self.repl.interact.confirm = lambda msg: True
        self.repl.interact.notify = lambda msg: None
        self.repl.config.editor = 'true'

    def test_create_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'newdir' / 'config'
            self.repl.config.config_path = config_path
            self.repl.edit_config()
            self.assertTrue(config_path.exists())