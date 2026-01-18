from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Sequence, TypeVar, Union
from .extensions import Extension
from .typing import Headers
@dataclass(frozen=True)
class RejectConnection(Event):
    """The rejection of a Websocket upgrade request, the HTTP response.

    The ``RejectConnection`` event sends the appropriate HTTP headers to
    communicate to the peer that the handshake has been rejected. You may also
    send an HTTP body by setting the ``has_body`` attribute to ``True`` and then
    sending one or more :class:`RejectData` events after this one. When sending
    a response body, the caller should set the ``Content-Length``,
    ``Content-Type``, and/or ``Transfer-Encoding`` headers as appropriate.

    When receiving a ``RejectConnection`` event, the ``has_body`` attribute will
    in almost all cases be ``True`` (even if the server set it to ``False``) and
    will be followed by at least one ``RejectData`` events, even though the data
    itself might be just ``b""``. (The only scenario in which the caller
    receives a ``RejectConnection`` with ``has_body == False`` is if the peer
    violates sends an informational status code (1xx) other than 101.)

    The ``has_body`` attribute should only be used when receiving the event. (It
    has ) is False the headers must include a
    content-length or transfer encoding.

    Fields:

    .. attribute:: headers (Headers)

       The headers to send with the response.

    .. attribute:: has_body

       This defaults to False, but set to True if there is a body. See
       also :class:`~RejectData`.

    .. attribute:: status_code

       The response status code.

    """
    status_code: int = 400
    headers: Headers = field(default_factory=list)
    has_body: bool = False