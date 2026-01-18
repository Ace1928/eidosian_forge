import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _on_basic_return(self, reply_code, reply_text, exchange, routing_key, message):
    """Return a failed message.

        This method returns an undeliverable message that was
        published with the "immediate" flag set, or an unroutable
        message published with the "mandatory" flag set. The reply
        code and text provide information about the reason that the
        message was undeliverable.

        PARAMETERS:
            reply_code: short

                The reply code. The AMQ reply codes are defined in AMQ
                RFC 011.

            reply_text: shortstr

                The localised reply text.  This text can be logged as an
                aid to resolving issues.

            exchange: shortstr

                Specifies the name of the exchange that the message
                was originally published to.

            routing_key: shortstr

                Message routing key

                Specifies the routing key name specified when the
                message was published.
        """
    exc = error_for_code(reply_code, reply_text, spec.Basic.Return, ChannelError)
    handlers = self.events.get('basic_return')
    if not handlers:
        raise exc
    for callback in handlers:
        callback(exc, exchange, routing_key, message)