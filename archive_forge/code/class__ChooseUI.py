import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
class _ChooseUI:
    """ Helper class for choose implementation.
    """

    def __init__(self, ui, msg, choices, default):
        self.ui = ui
        self._setup_mode()
        self._build_alternatives(msg, choices, default)

    def _setup_mode(self):
        """Setup input mode (line-based, char-based) and echo-back.

        Line-based input is used if the BRZ_TEXTUI_INPUT environment
        variable is set to 'line-based', or if there is no controlling
        terminal.
        """
        is_tty = self.ui.raw_stdin.isatty()
        if os.environ.get('BRZ_TEXTUI_INPUT') != 'line-based' and self.ui.raw_stdin == _unwrap_stream(sys.stdin) and is_tty:
            self.line_based = False
            self.echo_back = True
        else:
            self.line_based = True
            self.echo_back = not is_tty

    def _build_alternatives(self, msg, choices, default):
        """Parse choices string.

        Setup final prompt and the lists of choices and associated
        shortcuts.
        """
        index = 0
        help_list = []
        self.alternatives = {}
        choices = choices.split('\n')
        if default is not None and default not in range(0, len(choices)):
            raise ValueError('invalid default index')
        for c in choices:
            name = c.replace('&', '').lower()
            choice = (name, index)
            if name in self.alternatives:
                raise ValueError('duplicated choice: %s' % name)
            self.alternatives[name] = choice
            shortcut = c.find('&')
            if -1 != shortcut and shortcut + 1 < len(c):
                help = c[:shortcut]
                help += '[' + c[shortcut + 1] + ']'
                help += c[shortcut + 2:]
                shortcut = c[shortcut + 1]
            else:
                c = c.replace('&', '')
                shortcut = c[0]
                help = '[{}]{}'.format(shortcut, c[1:])
            shortcut = shortcut.lower()
            if shortcut in self.alternatives:
                raise ValueError('duplicated shortcut: %s' % shortcut)
            self.alternatives[shortcut] = choice
            if index == default:
                self.alternatives[''] = choice
                self.alternatives['\r'] = choice
            help_list.append(help)
            index += 1
        self.prompt = '{} ({}): '.format(msg, ', '.join(help_list))

    def _getline(self):
        line = self.ui.stdin.readline()
        if '' == line:
            raise EOFError
        return line.strip()

    def _getchar(self):
        char = osutils.getchar()
        if char == chr(3):
            raise KeyboardInterrupt
        if char == chr(4):
            raise EOFError
        if isinstance(char, bytes):
            return char.decode('ascii', 'replace')
        return char

    def interact(self):
        """Keep asking the user until a valid choice is made.
        """
        if self.line_based:
            getchoice = self._getline
        else:
            getchoice = self._getchar
        iter = 0
        while True:
            iter += 1
            if 1 == iter or self.line_based:
                self.ui.prompt(self.prompt)
            try:
                choice = getchoice()
            except EOFError:
                self.ui.stderr.write('\n')
                return None
            except KeyboardInterrupt:
                self.ui.stderr.write('\n')
                raise
            choice = choice.lower()
            if choice not in self.alternatives:
                continue
            name, index = self.alternatives[choice]
            if self.echo_back:
                self.ui.stderr.write(name + '\n')
            return index