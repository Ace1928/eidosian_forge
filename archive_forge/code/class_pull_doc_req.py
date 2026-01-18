from __future__ import annotations
import logging # isort:skip
from typing import Any
from ..message import Empty, Message
class pull_doc_req(Message[Empty]):
    """ Define the ``PULL-DOC-REQ`` message for requesting a Bokeh server reply
    with a new Bokeh Document.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'PULL-DOC-REQ'

    @classmethod
    def create(cls, **metadata: Any) -> pull_doc_req:
        """ Create an ``PULL-DOC-REQ`` message

        Any keyword arguments will be put into the message ``metadata``
        fragment as-is.

        """
        header = cls.create_header()
        return cls(header, metadata, Empty())