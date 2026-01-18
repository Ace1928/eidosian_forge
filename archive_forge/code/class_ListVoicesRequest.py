from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class ListVoicesRequest(proto.Message):
    """The top-level message sent by the client for the ``ListVoices``
    method.

    Attributes:
        language_code (str):
            Optional. Recommended.
            `BCP-47 <https://www.rfc-editor.org/rfc/bcp/bcp47.txt>`__
            language tag. If not specified, the API will return all
            supported voices. If specified, the ListVoices call will
            only return voices that can be used to synthesize this
            language_code. For example, if you specify ``"en-NZ"``, all
            ``"en-NZ"`` voices will be returned. If you specify
            ``"no"``, both ``"no-\\*"`` (Norwegian) and ``"nb-\\*"``
            (Norwegian Bokmal) voices will be returned.
    """
    language_code: str = proto.Field(proto.STRING, number=1)