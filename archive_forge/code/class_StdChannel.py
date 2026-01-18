from __future__ import annotations
import errno
import socket
from typing import TYPE_CHECKING
from amqp.exceptions import RecoverableConnectionError
from kombu.exceptions import ChannelError, ConnectionError
from kombu.message import Message
from kombu.utils.functional import dictfilter
from kombu.utils.objects import cached_property
from kombu.utils.time import maybe_s_to_ms
class StdChannel:
    """Standard channel base class."""
    no_ack_consumers = None

    def Consumer(self, *args, **kwargs):
        from kombu.messaging import Consumer
        return Consumer(self, *args, **kwargs)

    def Producer(self, *args, **kwargs):
        from kombu.messaging import Producer
        return Producer(self, *args, **kwargs)

    def get_bindings(self):
        raise _LeftBlank(self, 'get_bindings')

    def after_reply_message_received(self, queue):
        """Callback called after RPC reply received.

        Notes
        -----
           Reply queue semantics: can be used to delete the queue
           after transient reply message received.
        """

    def prepare_queue_arguments(self, arguments, **kwargs):
        return arguments

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        self.close()