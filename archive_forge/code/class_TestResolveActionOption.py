import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveActionOption(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.options = [conflicts.ResolveActionOption()]
        self.parser = option.get_optparser(self.options)

    def parse(self, args):
        return self.parser.parse_args(args)

    def test_unknown_action(self):
        self.assertRaises(option.BadOptionValue, self.parse, ['--action', 'take-me-to-the-moon'])

    def test_done(self):
        opts, args = self.parse(['--action', 'done'])
        self.assertEqual({'action': 'done'}, opts)

    def test_take_this(self):
        opts, args = self.parse(['--action', 'take-this'])
        self.assertEqual({'action': 'take_this'}, opts)
        opts, args = self.parse(['--take-this'])
        self.assertEqual({'action': 'take_this'}, opts)

    def test_take_other(self):
        opts, args = self.parse(['--action', 'take-other'])
        self.assertEqual({'action': 'take_other'}, opts)
        opts, args = self.parse(['--take-other'])
        self.assertEqual({'action': 'take_other'}, opts)