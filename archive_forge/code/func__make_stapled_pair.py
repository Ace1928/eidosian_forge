from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def _make_stapled_pair(one_way_pair: Callable[[], tuple[SendStreamT, ReceiveStreamT]]) -> tuple[StapledStream[SendStreamT, ReceiveStreamT], StapledStream[SendStreamT, ReceiveStreamT]]:
    pipe1_send, pipe1_recv = one_way_pair()
    pipe2_send, pipe2_recv = one_way_pair()
    stream1 = StapledStream(pipe1_send, pipe2_recv)
    stream2 = StapledStream(pipe2_send, pipe1_recv)
    return (stream1, stream2)