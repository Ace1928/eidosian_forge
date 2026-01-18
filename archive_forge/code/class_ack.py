from __future__ import annotations
import logging # isort:skip
from typing import Any
from ..message import Empty, Message
class ack(Message[Empty]):
    """ Define the ``ACK`` message for acknowledging successful client
    connection to a Bokeh server.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'ACK'

    @classmethod
    def create(cls, **metadata: Any) -> ack:
        """ Create an ``ACK`` message

        Any keyword arguments will be put into the message ``metadata``
        fragment as-is.

        """
        header = cls.create_header()
        content = Empty()
        return cls(header, metadata, content)