import unicodedata
from wcwidth import wcwidth
from IPython.core.completer import (
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
import pygments.lexers as pygments_lexers
import os
import sys
import traceback
class IPythonPTCompleter(Completer):
    """Adaptor to provide IPython completions to prompt_toolkit"""

    def __init__(self, ipy_completer=None, shell=None):
        if shell is None and ipy_completer is None:
            raise TypeError('Please pass shell=an InteractiveShell instance.')
        self._ipy_completer = ipy_completer
        self.shell = shell

    @property
    def ipy_completer(self):
        if self._ipy_completer:
            return self._ipy_completer
        else:
            return self.shell.Completer

    def get_completions(self, document, complete_event):
        if not document.current_line.strip():
            return
        with patch_stdout(), provisionalcompleter():
            body = document.text
            cursor_row = document.cursor_position_row
            cursor_col = document.cursor_position_col
            cursor_position = document.cursor_position
            offset = cursor_to_position(body, cursor_row, cursor_col)
            try:
                yield from self._get_completions(body, offset, cursor_position, self.ipy_completer)
            except Exception as e:
                try:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_tb)
                except AttributeError:
                    print('Unrecoverable Error in completions')

    @staticmethod
    def _get_completions(body, offset, cursor_position, ipyc):
        """
        Private equivalent of get_completions() use only for unit_testing.
        """
        debug = getattr(ipyc, 'debug', False)
        completions = _deduplicate_completions(body, ipyc.completions(body, offset))
        for c in completions:
            if not c.text:
                continue
            text = unicodedata.normalize('NFC', c.text)
            if wcwidth(text[0]) == 0:
                if cursor_position + c.start > 0:
                    char_before = body[c.start - 1]
                    fixed_text = unicodedata.normalize('NFC', char_before + text)
                    if wcwidth(text[0:1]) == 1:
                        yield Completion(fixed_text, start_position=c.start - offset - 1)
                        continue
            display_text = c.text
            adjusted_text = _adjust_completion_text_based_on_context(c.text, body, offset)
            if c.type == 'function':
                yield Completion(adjusted_text, start_position=c.start - offset, display=_elide(display_text + '()', body[c.start:c.end]), display_meta=c.type + c.signature)
            else:
                yield Completion(adjusted_text, start_position=c.start - offset, display=_elide(display_text, body[c.start:c.end]), display_meta=c.type)