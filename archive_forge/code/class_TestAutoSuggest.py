import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
class TestAutoSuggest(unittest.TestCase):

    def test_changing_provider(self):
        ip = get_ipython()
        ip.autosuggestions_provider = None
        self.assertEqual(ip.auto_suggest, None)
        ip.autosuggestions_provider = 'AutoSuggestFromHistory'
        self.assertIsInstance(ip.auto_suggest, AutoSuggestFromHistory)
        ip.autosuggestions_provider = 'NavigableAutoSuggestFromHistory'
        self.assertIsInstance(ip.auto_suggest, NavigableAutoSuggestFromHistory)