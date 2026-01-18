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
def _display_history(config, script, base, head, currents=()):
    for sc in script.walk_revisions(base=base or 'base', head=head or 'heads'):
        if indicate_current:
            sc._db_current_indicator = sc.revision in currents
        config.print_stdout(sc.cmd_format(verbose=verbose, include_branches=True, include_doc=True, include_parents=True))