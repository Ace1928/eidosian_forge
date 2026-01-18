from __future__ import unicode_literals
from prompt_toolkit.buffer import EditReadOnlyBuffer
from prompt_toolkit.filters.cli import ViNavigationMode
from prompt_toolkit.keys import Keys, Key
from prompt_toolkit.utils import Event
from .registry import BaseRegistry
from collections import deque
from six.moves import range
import weakref
import six
def _get_matches(self, key_presses):
    """
        For a list of :class:`KeyPress` instances. Give the matching handlers
        that would handle this.
        """
    keys = tuple((k.key for k in key_presses))
    cli = self._cli_ref()
    return [b for b in self._registry.get_bindings_for_keys(keys) if b.filter(cli)]