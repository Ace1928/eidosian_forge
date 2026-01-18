import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _setup_listeners(self):
    self._callbacks.update({spec.Channel.Close: self._on_close, spec.Channel.CloseOk: self._on_close_ok, spec.Channel.Flow: self._on_flow, spec.Channel.OpenOk: self._on_open_ok, spec.Basic.Cancel: self._on_basic_cancel, spec.Basic.CancelOk: self._on_basic_cancel_ok, spec.Basic.Deliver: self._on_basic_deliver, spec.Basic.Return: self._on_basic_return, spec.Basic.Ack: self._on_basic_ack, spec.Basic.Nack: self._on_basic_nack})