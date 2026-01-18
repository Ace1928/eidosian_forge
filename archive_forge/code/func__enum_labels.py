import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def _enum_labels(value: typing.Union[int, str, enum.Enum], enum_type: typing.Optional[typing.Type]=None) -> typing.Dict[int, str]:
    """Gets the human friendly labels of a known enum and what value they map to."""

    def get_labels(v: typing.Any) -> typing.Dict[int, str]:
        return typing.cast(typing.Dict[int, str], getattr(v, 'native_labels', lambda: {})())
    return get_labels(enum_type) if enum_type else get_labels(value)