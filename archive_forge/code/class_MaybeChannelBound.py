from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from .connection import maybe_channel
from .exceptions import NotBoundError
from .utils.functional import ChannelPromise
class MaybeChannelBound(Object):
    """Mixin for classes that can be bound to an AMQP channel."""
    _channel: Channel | None = None
    _is_bound = False
    can_cache_declaration = False

    def __call__(self: _MaybeChannelBoundType, channel: Channel | Connection) -> _MaybeChannelBoundType:
        """`self(channel) -> self.bind(channel)`."""
        return self.bind(channel)

    def bind(self: _MaybeChannelBoundType, channel: Channel | Connection) -> _MaybeChannelBoundType:
        """Create copy of the instance that is bound to a channel."""
        return copy(self).maybe_bind(channel)

    def maybe_bind(self: _MaybeChannelBoundType, channel: Channel | Connection) -> _MaybeChannelBoundType:
        """Bind instance to channel if not already bound."""
        if not self.is_bound and channel:
            self._channel = maybe_channel(channel)
            self.when_bound()
            self._is_bound = True
        return self

    def revive(self, channel: Channel) -> None:
        """Revive channel after the connection has been re-established.

        Used by :meth:`~kombu.Connection.ensure`.

        """
        if self.is_bound:
            self._channel = channel
            self.when_bound()

    def when_bound(self) -> None:
        """Callback called when the class is bound."""

    def __repr__(self) -> str:
        return self._repr_entity(type(self).__name__)

    def _repr_entity(self, item: str='') -> str:
        item = item or type(self).__name__
        if self.is_bound:
            return '<{} bound to chan:{}>'.format(item or type(self).__name__, self.channel.channel_id)
        return f'<unbound {item}>'

    @property
    def is_bound(self) -> bool:
        """Flag set if the channel is bound."""
        return self._is_bound and self._channel is not None

    @property
    def channel(self) -> Channel:
        """Current channel if the object is bound."""
        channel = self._channel
        if channel is None:
            raise NotBoundError("Can't call method on {} not bound to a channel".format(type(self).__name__))
        if isinstance(channel, ChannelPromise):
            channel = self._channel = channel()
        return channel