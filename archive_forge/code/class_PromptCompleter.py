from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
class PromptCompleter(object):
    """Prompt + input + completion.

  Yes, this is a roll-your own implementation.
  Yes, readline is that bad:
    linux: is unaware of the prompt even though it overrise raw_input()
    macos: different implementation than linux, and more brokener
    windows: didn't even try to implement
  """
    _CONTROL_C = '\x03'
    _DELETE = '\x7f'

    def __init__(self, prompt, choices=None, out=None, width=None, height=None, pad='  '):
        """Constructor.

    Args:
      prompt: str or None, The prompt string.
      choices: callable or list, A callable with no arguments that returns the
        list of all choices, or the list of choices.
      out: stream, The output stream, sys.stderr by default.
      width: int, The total display width in characters.
      height: int, The total display height in lines.
      pad: str, String inserted before each column.
    """
        self._prompt = prompt
        self._choices = choices
        self._out = out or sys.stderr
        self._attr = console_attr.ConsoleAttr()
        term_width, term_height = self._attr.GetTermSize()
        if width is None:
            width = 80
            if width > term_width:
                width = term_width
        self._width = width
        if height is None:
            height = 40
            if height > term_height:
                height = term_height
        self._height = height
        self._pad = pad

    def Input(self):
        """Reads and returns one line of user input with TAB complation."""
        all_choices = None
        matches = []
        response = []
        if self._prompt:
            self._out.write(self._prompt)
        c = None
        while True:
            previous_c = c
            c = self._attr.GetRawKey()
            if c in (None, '\n', '\r', PromptCompleter._CONTROL_C) or len(c) != 1:
                self._out.write('\n')
                break
            elif c in ('\x08', PromptCompleter._DELETE):
                if response:
                    response.pop()
                    self._out.write('\x08 \x08')
                    matches = all_choices
            elif c == '\t':
                response_prefix = ''.join(response)
                if previous_c == c:
                    matches = _PrefixMatches(response_prefix, matches)
                    if len(matches) > 1:
                        self._Display(response_prefix, matches)
                else:
                    if all_choices is None:
                        if callable(self._choices):
                            all_choices = self._choices()
                        else:
                            all_choices = self._choices
                    matches = all_choices
                    matches = _PrefixMatches(response_prefix, matches)
                    response_prefix = ''.join(response)
                    common_prefix = os.path.commonprefix(matches)
                    if len(common_prefix) > len(response):
                        matches = _PrefixMatches(common_prefix, matches)
                        self._out.write(common_prefix[len(response):])
                        response = list(common_prefix)
            else:
                response.append(c)
                self._out.write(c)
        return ''.join(response)

    def _Display(self, prefix, matches):
        """Displays the possible completions and redraws the prompt and response.

    Args:
      prefix: str, The current response.
      matches: [str], The list of strings that start with prefix.
    """
        row_data = _TransposeListToRows(matches, width=self._width, height=self._height, pad=self._pad, bold=self._attr.GetFontCode(bold=True), normal=self._attr.GetFontCode())
        if self._prompt:
            row_data.append(self._prompt)
        row_data.append(prefix)
        self._out.write(''.join(row_data))