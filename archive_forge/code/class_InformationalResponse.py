import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Tuple, Union
from ._abnf import method, request_target
from ._headers import Headers, normalize_and_validate
from ._util import bytesify, LocalProtocolError, validate
@dataclass(init=False, frozen=True)
class InformationalResponse(_ResponseBase):
    """An HTTP informational response.

    Fields:

    .. attribute:: status_code

       The status code of this response, as an integer. For an
       :class:`InformationalResponse`, this is always in the range [100,
       200).

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for
       details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    .. attribute:: reason

       The reason phrase of this response, as a byte string. For example:
       ``b"OK"``, or ``b"Not Found"``.

    """

    def __post_init__(self) -> None:
        if not 100 <= self.status_code < 200:
            raise LocalProtocolError('InformationalResponse status_code should be in range [100, 200), not {}'.format(self.status_code))
    __hash__ = None