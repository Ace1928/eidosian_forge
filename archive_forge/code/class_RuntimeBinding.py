import os
import signal
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Any, Optional, List
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.key_binding.bindings.completion import (
from prompt_toolkit.key_binding.vi_state import InputMode, ViState
from prompt_toolkit.filters import Condition
from IPython.core.getipython import get_ipython
from IPython.terminal.shortcuts import auto_match as match
from IPython.terminal.shortcuts import auto_suggest
from IPython.terminal.shortcuts.filters import filter_from_string
from IPython.utils.decorators import undoc
from prompt_toolkit.enums import DEFAULT_BUFFER
@dataclass
class RuntimeBinding(BaseBinding):
    filter: Condition