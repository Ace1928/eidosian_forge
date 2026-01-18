from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
class SuffixConsoleMessage(object):
    """A suffix-only implementation of ConsoleMessage."""

    def __init__(self, message, stream, suffix='', detail_message_callback=None, indentation_level=0):
        """Constructor.

    Args:
      message: str, the message that this object represents.
      stream: The output stream to write to.
      suffix: str, The suffix that will be appended to the very end of the
        message.
      detail_message_callback: func() -> str, A no argument function that will
        be called and the result will be added after the message and before the
        suffix on every call to Print().
      indentation_level: int, The indentation level of the message. Each
        indentation is represented by two spaces.
    """
        self._stream = stream
        self._message = message
        self._suffix = suffix
        self._console_width = console_attr.ConsoleAttr().GetTermSize()[0] - 1
        if self._console_width < 0:
            self._console_width = 0
        self._detail_message_callback = detail_message_callback
        self._level = indentation_level
        self._no_output = False
        if self._console_width - INDENTATION_WIDTH * indentation_level <= 0:
            self._no_output = True
        self._num_lines = 0
        self._lines = []
        self._has_printed = False

    def _UpdateSuffix(self, suffix):
        """Updates the suffix for this message."""
        if not isinstance(suffix, six.string_types):
            raise TypeError('expected a string or other character buffer object')
        self._suffix = suffix

    def Print(self, print_all=False):
        """Prints out the message to the console.

    The implementation of this function assumes that when called, the
    cursor position of the terminal is on the same line as the last line
    that this function printed (and nothing more). The exception for this is if
    this is the first time that print is being called on this message or if
    print_all is True. The implementation should also return the cursor to
    the last line of the printed message. The cursor position in this case
    should be at the end of printed text to avoid text being overwritten.

    Args:
      print_all: bool, if the entire message should be printed instead of just
        updating the message.
    """
        if self._console_width == 0 or self._no_output:
            return
        message = self.GetMessage()
        if not message:
            return
        if not self._has_printed or print_all:
            self._has_printed = True
            self._ClearLine()
            self._lines = self._SplitMessageIntoLines(message)
            self._num_lines = len(self._lines)
            for line in self._lines:
                self._WriteLine(line)
            return
        new_lines = self._SplitMessageIntoLines(message)
        new_num_lines = len(new_lines)
        if new_num_lines < self._num_lines:
            self._stream.write('\n')
            for line in new_lines:
                self._WriteLine(line)
        else:
            matching_lines = self._GetNumMatchingLines(new_lines)
            if self._num_lines - matching_lines <= 1:
                lines_to_print = new_num_lines - self._num_lines + 1
                self._ClearLine()
                for line in new_lines[-1 * lines_to_print:]:
                    self._WriteLine(line)
            else:
                self._stream.write('\n')
                for line in new_lines:
                    self._WriteLine(line)
        self._lines = new_lines
        self._num_lines = new_num_lines

    def GetMessage(self):
        if self._detail_message_callback:
            detail_message = self._detail_message_callback()
            if detail_message:
                return self._message + detail_message + self._suffix
        return self._message + self._suffix

    @property
    def effective_width(self):
        """The effective width when the indentation level is considered."""
        return self._console_width - INDENTATION_WIDTH * self._level

    def _GetNumMatchingLines(self, new_lines):
        matching_lines = 0
        for i in range(min(len(new_lines), self._num_lines)):
            if new_lines[i] != self._lines[i]:
                break
            matching_lines += 1
        return matching_lines

    def _SplitMessageIntoLines(self, message):
        """Converts message into a list of strs, each representing a line."""
        lines = []
        pos = 0
        while pos < len(message):
            lines.append(message[pos:pos + self.effective_width])
            pos += self.effective_width
            if pos < len(message):
                lines[-1] += '\n'
        return lines

    def _ClearLine(self):
        self._stream.write('\r{}\r'.format(' ' * self._console_width))

    def _WriteLine(self, line):
        self._stream.write(self._level * INDENTATION_WIDTH * ' ' + line)
        self._stream.flush()