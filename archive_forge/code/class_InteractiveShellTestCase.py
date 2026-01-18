import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory
class InteractiveShellTestCase(unittest.TestCase):

    def rl_hist_entries(self, rl, n):
        """Get last n readline history entries as a list"""
        return [rl.get_history_item(rl.get_current_history_length() - x) for x in range(n - 1, -1, -1)]

    @mock_input
    def test_inputtransformer_syntaxerror(self):
        ip = get_ipython()
        ip.input_transformers_post.append(syntax_error_transformer)
        try:
            with tt.AssertPrints('4', suppress=False):
                yield u'print(2*2)'
            with tt.AssertPrints('SyntaxError: input contains', suppress=False):
                yield u'print(2345) # syntaxerror'
            with tt.AssertPrints('16', suppress=False):
                yield u'print(4*4)'
        finally:
            ip.input_transformers_post.remove(syntax_error_transformer)

    def test_repl_not_plain_text(self):
        ip = get_ipython()
        formatter = ip.display_formatter
        assert formatter.active_types == ['text/plain']
        assert formatter.ipython_display_formatter.enabled

        class Test(object):

            def __repr__(self):
                return '<Test %i>' % id(self)

            def _repr_html_(self):
                return '<html>'
        obj = Test()
        data, _ = formatter.format(obj)
        self.assertEqual(data, {'text/plain': repr(obj)})

        class Test2(Test):

            def _ipython_display_(self):
                from IPython.display import display, HTML
                display(HTML('<custom>'))
        called = False

        def handler(data, metadata):
            print('Handler called')
            nonlocal called
            called = True
        ip.display_formatter.active_types.append('text/html')
        ip.display_formatter.formatters['text/html'].enabled = True
        ip.mime_renderers['text/html'] = handler
        try:
            obj = Test()
            display(obj)
        finally:
            ip.display_formatter.formatters['text/html'].enabled = False
            del ip.mime_renderers['text/html']
        assert called == True