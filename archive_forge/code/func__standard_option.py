import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _standard_option(name, **kwargs):
    """Register a standard option."""
    Option.STD_OPTIONS[name] = Option(name, **kwargs)
    Option.OPTIONS[name] = Option.STD_OPTIONS[name]