import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
def _register_last(self, event_name, handler, unique_id, unique_id_uses_count=False):
    self._register_section(event_name, handler, unique_id, unique_id_uses_count, section=_LAST)