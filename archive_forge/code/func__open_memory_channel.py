from __future__ import annotations
from collections import OrderedDict, deque
from math import inf
from typing import (
import attrs
from outcome import Error, Value
import trio
from ._abc import ReceiveChannel, ReceiveType, SendChannel, SendType, T
from ._core import Abort, RaiseCancelT, Task, enable_ki_protection
from ._util import NoPublicConstructor, final, generic_function
def _open_memory_channel(max_buffer_size: int | float) -> tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]:
    """Open a channel for passing objects between tasks within a process.

    Memory channels are lightweight, cheap to allocate, and entirely
    in-memory. They don't involve any operating-system resources, or any kind
    of serialization. They just pass Python objects directly between tasks
    (with a possible stop in an internal buffer along the way).

    Channel objects can be closed by calling `~trio.abc.AsyncResource.aclose`
    or using ``async with``. They are *not* automatically closed when garbage
    collected. Closing memory channels isn't mandatory, but it is generally a
    good idea, because it helps avoid situations where tasks get stuck waiting
    on a channel when there's no-one on the other side. See
    :ref:`channel-shutdown` for details.

    Memory channel operations are all atomic with respect to
    cancellation, either `~trio.abc.ReceiveChannel.receive` will
    successfully return an object, or it will raise :exc:`Cancelled`
    while leaving the channel unchanged.

    Args:
      max_buffer_size (int or math.inf): The maximum number of items that can
        be buffered in the channel before :meth:`~trio.abc.SendChannel.send`
        blocks. Choosing a sensible value here is important to ensure that
        backpressure is communicated promptly and avoid unnecessary latency;
        see :ref:`channel-buffering` for more details. If in doubt, use 0.

    Returns:
      A pair ``(send_channel, receive_channel)``. If you have
      trouble remembering which order these go in, remember: data
      flows from left → right.

    In addition to the standard channel methods, all memory channel objects
    provide a ``statistics()`` method, which returns an object with the
    following fields:

    * ``current_buffer_used``: The number of items currently stored in the
      channel buffer.
    * ``max_buffer_size``: The maximum number of items allowed in the buffer,
      as passed to :func:`open_memory_channel`.
    * ``open_send_channels``: The number of open
      :class:`MemorySendChannel` endpoints pointing to this channel.
      Initially 1, but can be increased by
      :meth:`MemorySendChannel.clone`.
    * ``open_receive_channels``: Likewise, but for open
      :class:`MemoryReceiveChannel` endpoints.
    * ``tasks_waiting_send``: The number of tasks blocked in ``send`` on this
      channel (summing over all clones).
    * ``tasks_waiting_receive``: The number of tasks blocked in ``receive`` on
      this channel (summing over all clones).

    """
    if max_buffer_size != inf and (not isinstance(max_buffer_size, int)):
        raise TypeError('max_buffer_size must be an integer or math.inf')
    if max_buffer_size < 0:
        raise ValueError('max_buffer_size must be >= 0')
    state: MemoryChannelState[T] = MemoryChannelState(max_buffer_size)
    return (MemorySendChannel[T]._create(state), MemoryReceiveChannel[T]._create(state))