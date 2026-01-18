from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
class MediaEvent:
    """Representation of a media event.

    These events are used internally by some audio driver implementation to
    communicate events to the :class:`~pyglet.media.player.Player`.
    One example is the ``on_eos`` event.

    Args:
        event (str): Event description.
        timestamp (float): The time when this event happens.
        *args: Any required positional argument to go along with this event.
    """
    __slots__ = ('event', 'timestamp', 'args')

    def __init__(self, event, timestamp=0.0, *args):
        self.event = event
        self.timestamp = timestamp
        self.args = args

    def sync_dispatch_to_player(self, player):
        pyglet.app.platform_event_loop.post_event(player, self.event, *self.args)

    def __repr__(self):
        return f'MediaEvent({self.event}, {self.timestamp}, {self.args})'

    def __lt__(self, other):
        if not isinstance(other, MediaEvent):
            return NotImplemented
        return self.timestamp < other.timestamp