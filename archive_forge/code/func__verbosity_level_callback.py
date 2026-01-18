import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _verbosity_level_callback(option, opt_str, value, parser):
    global _verbosity_level
    if not value:
        _verbosity_level = 0
    elif opt_str == 'verbose':
        if _verbosity_level > 0:
            _verbosity_level += 1
        else:
            _verbosity_level = 1
    elif _verbosity_level < 0:
        _verbosity_level -= 1
    else:
        _verbosity_level = -1