import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def _load_known_opts(self, optfile, parsed):
    """
        Pull in CLI args for proper models/tasks/etc.

        Called before args are parsed; ``_load_opts`` is used for actually overriding
        opts after they are parsed.
        """
    new_opt = Opt.load(optfile)
    for key, value in new_opt.items():
        if key not in parsed or parsed[key] is None:
            parsed[key] = value