from __future__ import annotations
import numbers
from .abstract import MaybeChannelBound, Object
from .exceptions import ContentDisallowed
from .serialization import prepare_accept_content
def _can_declare(self):
    return not self.no_declare and (self.name and (not self.name.startswith(INTERNAL_EXCHANGE_PREFIX)))