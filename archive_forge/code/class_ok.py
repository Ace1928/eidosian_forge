from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.types import ID
from ..message import Empty, Message
class ok(Message[Empty]):
    """ Define the ``OK`` message for acknowledging successful handling of a
    previous message.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'OK'

    @classmethod
    def create(cls, request_id: ID, **metadata: Any) -> ok:
        """ Create an ``OK`` message

        Args:
            request_id (str) :
                The message ID for the message the precipitated the OK.

        Any additional keyword arguments will be put into the message
        ``metadata`` fragment as-is.

        """
        header = cls.create_header(request_id=request_id)
        return cls(header, metadata, Empty())