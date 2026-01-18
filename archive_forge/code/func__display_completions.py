import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _display_completions(self, substitution, matches, max_length):
    """
        Display the completions. Invoked by readline.
        @param substitution: string to complete
        @param matches: list of possible matches
        @param max_length: length of the longest matching item
        """
    x_orig = self.con.get_cursor_xy()[0]
    width = self.con.get_width()
    max_length += 2

    def just(text):
        """
            Justifies the text to the max match length.
            """
        return text.ljust(max_length, ' ')
    if self._current_parameter:
        keywords = []
        values = []
        for match in matches:
            if match.endswith('='):
                keywords.append(self.con.render_text(just(match), self.prefs['color_keyword']))
            elif '=' in match:
                _, _, value = match.partition('=')
                values.append(self.con.render_text(just(value), self.prefs['color_parameter']))
            else:
                values.append(self.con.render_text(just(match), self.prefs['color_parameter']))
        matches = values + keywords
    else:
        paths = []
        commands = []
        for match in matches:
            if '/' in match or match.startswith('@') or '*' in match:
                paths.append(self.con.render_text(just(match), self.prefs['color_path']))
            else:
                commands.append(self.con.render_text(just(match), self.prefs['color_command']))
        matches = paths + commands
    self.con.raw_write('\n')
    if matches:
        if max_length < width:
            nr_cols = width // max_length
        else:
            nr_cols = 1
        for i in six.moves.range(0, len(matches), nr_cols):
            self.con.raw_write(''.join(matches[i:i + nr_cols]))
            self.con.raw_write('\n')
    line = '%s%s' % (self._get_prompt(), readline.get_line_buffer())
    self.con.raw_write('%s' % line)
    y_pos = self.con.get_cursor_xy()[1]
    self.con.set_cursor_xy(x_orig, y_pos)