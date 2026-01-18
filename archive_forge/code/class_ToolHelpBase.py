import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
class ToolHelpBase(ToolBase):
    description = 'Print tool list, shortcuts and description'
    default_keymap = property(lambda self: mpl.rcParams['keymap.help'])
    image = 'help'

    @staticmethod
    def format_shortcut(key_sequence):
        """
        Convert a shortcut string from the notation used in rc config to the
        standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.
        """
        return key_sequence if len(key_sequence) == 1 else re.sub('\\+[A-Z]', '+Shift\\g<0>', key_sequence).title()

    def _format_tool_keymap(self, name):
        keymaps = self.toolmanager.get_tool_keymap(name)
        return ', '.join((self.format_shortcut(keymap) for keymap in keymaps))

    def _get_help_entries(self):
        return [(name, self._format_tool_keymap(name), tool.description) for name, tool in sorted(self.toolmanager.tools.items()) if tool.description]

    def _get_help_text(self):
        entries = self._get_help_entries()
        entries = ['{}: {}\n\t{}'.format(*entry) for entry in entries]
        return '\n'.join(entries)

    def _get_help_html(self):
        fmt = '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'
        rows = [fmt.format('<b>Action</b>', '<b>Shortcuts</b>', '<b>Description</b>')]
        rows += [fmt.format(*row) for row in self._get_help_entries()]
        return '<style>td {padding: 0px 4px}</style><table><thead>' + rows[0] + '</thead><tbody>'.join(rows[1:]) + '</tbody></table>'