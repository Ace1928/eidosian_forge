import os
import sys
import inspect
from warnings import warn
from typing import Union as UnionType, Optional
from IPython.core.async_helpers import get_asyncio_loop
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.utils.py3compat import input
from IPython.utils.terminal import toggle_set_term_title, set_term_title, restore_term_title
from IPython.utils.process import abbrev_cwd
from traitlets import (
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import HasFocus, Condition, IsDone
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import History
from prompt_toolkit.layout.processors import ConditionalProcessor, HighlightMatchingBracketProcessor
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession, CompleteStyle, print_formatted_text
from prompt_toolkit.styles import DynamicStyle, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_cls, style_from_pygments_dict
from prompt_toolkit import __version__ as ptk_version
from pygments.styles import get_style_by_name
from pygments.style import Style
from pygments.token import Token
from .debugger import TerminalPdb, Pdb
from .magics import TerminalMagics
from .pt_inputhooks import get_inputhook_name_and_func
from .prompts import Prompts, ClassicPrompts, RichPromptDisplayHook
from .ptutils import IPythonPTCompleter, IPythonPTLexer
from .shortcuts import (
from .shortcuts.filters import KEYBINDING_FILTERS, filter_from_string
from .shortcuts.auto_suggest import (
def _merge_shortcuts(self, user_shortcuts):
    key_bindings = create_ipython_shortcuts(self)
    known_commands = {create_identifier(binding.command): binding.command for binding in KEY_BINDINGS}
    shortcuts_to_skip = []
    shortcuts_to_add = []
    for shortcut in user_shortcuts:
        command_id = shortcut['command']
        if command_id not in known_commands:
            allowed_commands = '\n - '.join(known_commands)
            raise ValueError(f'{command_id} is not a known shortcut command. Allowed commands are: \n - {allowed_commands}')
        old_keys = shortcut.get('match_keys', None)
        old_filter = filter_from_string(shortcut['match_filter']) if 'match_filter' in shortcut else None
        matching = [binding for binding in KEY_BINDINGS if (old_filter is None or binding.filter == old_filter) and (old_keys is None or [k for k in binding.keys] == old_keys) and (create_identifier(binding.command) == command_id)]
        new_keys = shortcut.get('new_keys', None)
        new_filter = shortcut.get('new_filter', None)
        command = known_commands[command_id]
        creating_new = shortcut.get('create', False)
        modifying_existing = not creating_new and (new_keys is not None or new_filter)
        if creating_new and new_keys == []:
            raise ValueError('Cannot add a shortcut without keys')
        if modifying_existing:
            specification = {key: shortcut[key] for key in ['command', 'filter'] if key in shortcut}
            if len(matching) == 0:
                raise ValueError(f'No shortcuts matching {specification} found in {KEY_BINDINGS}')
            elif len(matching) > 1:
                raise ValueError(f'Multiple shortcuts matching {specification} found, please add keys/filter to select one of: {matching}')
            matched = matching[0]
            old_filter = matched.filter
            old_keys = list(matched.keys)
            shortcuts_to_skip.append(RuntimeBinding(command, keys=old_keys, filter=old_filter))
        if new_keys != []:
            shortcuts_to_add.append(RuntimeBinding(command, keys=new_keys or old_keys, filter=filter_from_string(new_filter) if new_filter is not None else old_filter if old_filter is not None else filter_from_string('always')))
    key_bindings = create_ipython_shortcuts(self, skip=shortcuts_to_skip)
    for binding in shortcuts_to_add:
        add_binding(key_bindings, binding)
    return key_bindings