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
class IOVUnwrapResult(typing.NamedTuple):
    """Result of the `unwrap_iov()` function."""
    buffers: typing.Tuple[IOVResBuffer, ...]
    encrypted: bool
    qop: int