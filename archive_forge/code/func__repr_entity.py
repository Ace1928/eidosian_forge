from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from .connection import maybe_channel
from .exceptions import NotBoundError
from .utils.functional import ChannelPromise
def _repr_entity(self, item: str='') -> str:
    item = item or type(self).__name__
    if self.is_bound:
        return '<{} bound to chan:{}>'.format(item or type(self).__name__, self.channel.channel_id)
    return f'<unbound {item}>'