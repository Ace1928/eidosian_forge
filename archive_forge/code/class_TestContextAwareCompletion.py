import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
class TestContextAwareCompletion(unittest.TestCase):

    def test_adjust_completion_text_based_on_context(self):
        self.assertEqual(_adjust_completion_text_based_on_context('arg1=', 'func1(a=)', 7), 'arg1')
        self.assertEqual(_adjust_completion_text_based_on_context('arg1=', 'func1(a)', 7), 'arg1=')
        self.assertEqual(_adjust_completion_text_based_on_context('arg1=', 'func1(a', 7), 'arg1=')
        self.assertEqual(_adjust_completion_text_based_on_context('%magic', 'func1(a=)', 7), '%magic')
        self.assertEqual(_adjust_completion_text_based_on_context('func2', 'func1(a=)', 7), 'func2')