from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, TypedDict
from ..exceptions import ProtocolError
from ..message import Message
class push_doc(Message[PushDoc]):
    """ Define the ``PUSH-DOC`` message for pushing Documents from clients to a
    Bokeh server.

    The ``content`` fragment of for this message is has the form:

    .. code-block:: python

        {
            'doc' : <Document JSON>
        }

    """
    msgtype = 'PUSH-DOC'

    @classmethod
    def create(cls, document: Document, **metadata: Any) -> push_doc:
        """

        """
        header = cls.create_header()
        content = PushDoc(doc=document.to_json())
        msg = cls(header, metadata, content)
        return msg

    def push_to_document(self, doc: Document) -> None:
        """

        Raises:
            ProtocolError

        """
        if 'doc' not in self.content:
            raise ProtocolError('No doc in PUSH-DOC')
        doc.replace_with_json(self.content['doc'])