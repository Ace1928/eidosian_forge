from __future__ import annotations
import os
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from . import autogenerate as autogen
from . import util
from .runtime.environment import EnvironmentContext
from .script import ScriptDirectory
def _display_history_w_current(config, script, base, head):

    def _display_current_history(rev, context):
        if head == 'current':
            _display_history(config, script, base, rev, rev)
        elif base == 'current':
            _display_history(config, script, rev, head, rev)
        else:
            _display_history(config, script, base, head, rev)
        return []
    with EnvironmentContext(config, script, fn=_display_current_history):
        script.run_env()