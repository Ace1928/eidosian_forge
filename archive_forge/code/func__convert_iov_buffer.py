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
def _convert_iov_buffer(self, buffer: IOVBuffer) -> sspilib.raw.SecBuffer:
    data = bytearray()
    if isinstance(buffer.data, bytes):
        data = bytearray(buffer.data)
    elif isinstance(buffer.data, int) and (not isinstance(buffer.data, bool)):
        data = bytearray(buffer.data)
    else:
        auto_alloc_size = {BufferType.header: self._security_trailer, BufferType.padding: self._block_size, BufferType.trailer: self._security_trailer}
        alloc = buffer.data
        if alloc is None:
            alloc = buffer.type in auto_alloc_size
        if alloc:
            if buffer.type not in auto_alloc_size:
                raise ValueError('Cannot auto allocate buffer of type %s.%s' % (type(buffer.type).__name__, buffer.type.name))
            data = bytearray(auto_alloc_size[buffer.type])
    buffer_type = int(buffer.type)
    buffer_flags = 0
    if buffer_type == BufferType.sign_only:
        buffer_type = sspilib.raw.SecBufferType.SECBUFFER_DATA
        buffer_flags = sspilib.raw.SecBufferFlags.SECBUFFER_READONLY_WITH_CHECKSUM
    elif buffer_type == BufferType.data_readonly:
        buffer_type = sspilib.raw.SecBufferType.SECBUFFER_DATA
        buffer_flags = sspilib.raw.SecBufferFlags.SECBUFFER_READONLY
    return sspilib.raw.SecBuffer(data, buffer_type, buffer_flags)