import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _global_option(name, **kwargs):
    """Register a global option."""
    Option.OPTIONS[name] = Option(name, **kwargs)