from __future__ import annotations
import base64
import collections.abc
import logging
import os
import typing as t
from spnego._context import (
from spnego._credential import Credential, CredentialCache, Password, unify_credentials
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.exceptions import WinError as NativeError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
def _create_iov_result(iov: sspilib.raw.SecBufferDesc) -> tuple[IOVResBuffer, ...]:
    """Converts SSPI IOV buffer to generic IOVBuffer result."""
    buffers = []
    for i in iov:
        buffer_type = int(i.buffer_type)
        if i.buffer_flags & sspilib.raw.SecBufferFlags.SECBUFFER_READONLY_WITH_CHECKSUM:
            buffer_type = BufferType.sign_only
        elif i.buffer_flags & sspilib.raw.SecBufferFlags.SECBUFFER_READONLY:
            buffer_type = BufferType.data_readonly
        buffer_entry = IOVResBuffer(type=BufferType(buffer_type), data=i.data)
        buffers.append(buffer_entry)
    return tuple(buffers)