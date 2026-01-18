import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def _parse_merge_type(typestring):
    return get_merge_type(typestring)