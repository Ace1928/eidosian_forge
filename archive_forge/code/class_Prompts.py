from pygments.token import Token
import sys
from IPython.core.displayhook import DisplayHook
from prompt_toolkit.formatted_text import fragment_list_width, PygmentsTokens
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.enums import EditingMode
class Prompts(object):

    def __init__(self, shell):
        self.shell = shell

    def vi_mode(self):
        if getattr(self.shell.pt_app, 'editing_mode', None) == EditingMode.VI and self.shell.prompt_includes_vi_mode:
            mode = str(self.shell.pt_app.app.vi_state.input_mode)
            if mode.startswith('InputMode.'):
                mode = mode[10:13].lower()
            elif mode.startswith('vi-'):
                mode = mode[3:6]
            return '[' + mode + '] '
        return ''

    def current_line(self) -> int:
        if self.shell.pt_app is not None:
            return self.shell.pt_app.default_buffer.document.cursor_position_row or 0
        return 0

    def in_prompt_tokens(self):
        return [(Token.Prompt, self.vi_mode()), (Token.Prompt, self.shell.prompt_line_number_format.format(line=1, rel_line=-self.current_line())), (Token.Prompt, 'In ['), (Token.PromptNum, str(self.shell.execution_count)), (Token.Prompt, ']: ')]

    def _width(self):
        return fragment_list_width(self.in_prompt_tokens())

    def continuation_prompt_tokens(self, width=None, *, lineno=None):
        if width is None:
            width = self._width()
        line = lineno + 1 if lineno is not None else 0
        prefix = ' ' * len(self.vi_mode()) + self.shell.prompt_line_number_format.format(line=line, rel_line=line - self.current_line() - 1)
        return [(Token.Prompt, prefix + ' ' * (width - len(prefix) - 5) + '...: ')]

    def rewrite_prompt_tokens(self):
        width = self._width()
        return [(Token.Prompt, '-' * (width - 2) + '> ')]

    def out_prompt_tokens(self):
        return [(Token.OutPrompt, 'Out['), (Token.OutPromptNum, str(self.shell.execution_count)), (Token.OutPrompt, ']: ')]