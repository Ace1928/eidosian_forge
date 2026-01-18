import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class TransportHooks(hooks.Hooks):
    """Mapping of hook names to registered callbacks for transport hooks"""

    def __init__(self):
        super().__init__()
        self.add_hook('post_connect', 'Called after a new connection is established or a reconnect occurs. The sole argument passed is either the connected transport or smart medium instance.', (2, 5))