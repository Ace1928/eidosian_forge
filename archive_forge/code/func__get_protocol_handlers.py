import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _get_protocol_handlers():
    """Return a dictionary of {urlprefix: [factory]}"""
    return transport_list_registry