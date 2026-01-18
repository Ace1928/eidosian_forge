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
class ContextReq(enum.IntFlag):
    none = 0
    delegate = 1
    mutual_auth = 2
    replay_detect = 4
    sequence_detect = 8
    confidentiality = 16
    integrity = 32
    dce_style = 4096
    identify = 8192
    delegate_policy = 524288
    no_integrity = 268435456
    default = 2 | 4 | 8 | 16 | 32