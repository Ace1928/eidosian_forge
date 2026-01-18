from __future__ import unicode_literals
from .enums import DEFAULT_BUFFER, SEARCH_BUFFER, SYSTEM_BUFFER, DUMMY_BUFFER
from .buffer import Buffer, AcceptAction
from .history import InMemoryHistory
import six

        Pop buffer from the focus stack.
        