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
def _make_style_from_name_or_cls(self, name_or_cls):
    """
        Small wrapper that make an IPython compatible style from a style name

        We need that to add style for prompt ... etc.
        """
    style_overrides = {}
    if name_or_cls == 'legacy':
        legacy = self.colors.lower()
        if legacy == 'linux':
            style_cls = get_style_by_name('monokai')
            style_overrides = _style_overrides_linux
        elif legacy == 'lightbg':
            style_overrides = _style_overrides_light_bg
            style_cls = get_style_by_name('pastie')
        elif legacy == 'neutral':
            style_cls = get_style_by_name('default')
            style_overrides.update({Token.Number: '#ansigreen', Token.Operator: 'noinherit', Token.String: '#ansiyellow', Token.Name.Function: '#ansiblue', Token.Name.Class: 'bold #ansiblue', Token.Name.Namespace: 'bold #ansiblue', Token.Name.Variable.Magic: '#ansiblue', Token.Prompt: '#ansigreen', Token.PromptNum: '#ansibrightgreen bold', Token.OutPrompt: '#ansired', Token.OutPromptNum: '#ansibrightred bold'})
            if os.name == 'nt':
                style_overrides.update({Token.Prompt: '#ansidarkgreen', Token.PromptNum: '#ansigreen bold', Token.OutPrompt: '#ansidarkred', Token.OutPromptNum: '#ansired bold'})
        elif legacy == 'nocolor':
            style_cls = _NoStyle
            style_overrides = {}
        else:
            raise ValueError('Got unknown colors: ', legacy)
    else:
        if isinstance(name_or_cls, str):
            style_cls = get_style_by_name(name_or_cls)
        else:
            style_cls = name_or_cls
        style_overrides = {Token.Prompt: '#ansigreen', Token.PromptNum: '#ansibrightgreen bold', Token.OutPrompt: '#ansired', Token.OutPromptNum: '#ansibrightred bold'}
    style_overrides.update(self.highlighting_style_overrides)
    style = merge_styles([style_from_pygments_cls(style_cls), style_from_pygments_dict(style_overrides)])
    return style