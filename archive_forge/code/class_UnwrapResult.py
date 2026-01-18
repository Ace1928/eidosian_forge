import abc
import dataclasses
import enum
import typing
import warnings
from spnego._credential import Credential
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import FeatureMissingError, NegotiateOptions, SpnegoError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
class UnwrapResult(typing.NamedTuple):
    """Result of the `unwrap()` function."""
    data: bytes
    encrypted: bool
    qop: int