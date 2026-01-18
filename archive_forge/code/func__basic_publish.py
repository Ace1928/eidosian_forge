import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _basic_publish(self, msg, exchange='', routing_key='', mandatory=False, immediate=False, timeout=None, confirm_timeout=None, argsig='Bssbb'):
    """Publish a message.

        This method publishes a message to a specific exchange. The
        message will be routed to queues as defined by the exchange
        configuration and distributed to any active consumers when the
        transaction, if any, is committed.

        When channel is in confirm mode (when Connection parameter
        confirm_publish is set to True), each message is confirmed.
        When broker rejects published message (e.g. due internal broker
        constrains), MessageNacked exception is raised and
        set confirm_timeout to wait maximum confirm_timeout second
        for message to confirm.

        PARAMETERS:
            exchange: shortstr

                Specifies the name of the exchange to publish to.  The
                exchange name can be empty, meaning the default
                exchange.  If the exchange name is specified, and that
                exchange does not exist, the server will raise a
                channel exception.

                RULE:

                    The server MUST accept a blank exchange name to
                    mean the default exchange.

                RULE:

                    The exchange MAY refuse basic content in which
                    case it MUST raise a channel exception with reply
                    code 540 (not implemented).

            routing_key: shortstr

                Message routing key

                Specifies the routing key for the message.  The
                routing key is used for routing messages depending on
                the exchange configuration.

            mandatory: boolean

                indicate mandatory routing

                This flag tells the server how to react if the message
                cannot be routed to a queue.  If this flag is True, the
                server will return an unroutable message with a Return
                method.  If this flag is False, the server silently
                drops the message.

                RULE:

                    The server SHOULD implement the mandatory flag.

            immediate: boolean

                request immediate delivery

                This flag tells the server how to react if the message
                cannot be routed to a queue consumer immediately.  If
                this flag is set, the server will return an
                undeliverable message with a Return method. If this
                flag is zero, the server will queue the message, but
                with no guarantee that it will ever be consumed.

                RULE:

                    The server SHOULD implement the immediate flag.

            timeout: short

                timeout for publish

                Set timeout to wait maximum timeout second
                for message to publish.

            confirm_timeout: short

                confirm_timeout for publish in confirm mode

                When the channel is in confirm mode set
                confirm_timeout to wait maximum confirm_timeout
                second for message to confirm.

        """
    if not self.connection:
        raise RecoverableConnectionError('basic_publish: connection closed')
    capabilities = self.connection.client_properties.get('capabilities', {})
    if capabilities.get('connection.blocked', False):
        try:
            self.connection.drain_events(timeout=0)
        except socket.timeout:
            pass
    try:
        with self.connection.transport.having_timeout(timeout):
            return self.send_method(spec.Basic.Publish, argsig, (0, exchange, routing_key, mandatory, immediate), msg)
    except socket.timeout:
        raise RecoverableChannelError('basic_publish: timed out')