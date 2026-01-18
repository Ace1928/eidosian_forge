from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
class PerMessageDeflate(Extension):
    """
    Per-Message Deflate extension.

    """
    name = ExtensionName('permessage-deflate')

    def __init__(self, remote_no_context_takeover: bool, local_no_context_takeover: bool, remote_max_window_bits: int, local_max_window_bits: int, compress_settings: Optional[Dict[Any, Any]]=None) -> None:
        """
        Configure the Per-Message Deflate extension.

        """
        if compress_settings is None:
            compress_settings = {}
        assert remote_no_context_takeover in [False, True]
        assert local_no_context_takeover in [False, True]
        assert 8 <= remote_max_window_bits <= 15
        assert 8 <= local_max_window_bits <= 15
        assert 'wbits' not in compress_settings
        self.remote_no_context_takeover = remote_no_context_takeover
        self.local_no_context_takeover = local_no_context_takeover
        self.remote_max_window_bits = remote_max_window_bits
        self.local_max_window_bits = local_max_window_bits
        self.compress_settings = compress_settings
        if not self.remote_no_context_takeover:
            self.decoder = zlib.decompressobj(wbits=-self.remote_max_window_bits)
        if not self.local_no_context_takeover:
            self.encoder = zlib.compressobj(wbits=-self.local_max_window_bits, **self.compress_settings)
        self.decode_cont_data = False

    def __repr__(self) -> str:
        return f'PerMessageDeflate(remote_no_context_takeover={self.remote_no_context_takeover}, local_no_context_takeover={self.local_no_context_takeover}, remote_max_window_bits={self.remote_max_window_bits}, local_max_window_bits={self.local_max_window_bits})'

    def decode(self, frame: frames.Frame, *, max_size: Optional[int]=None) -> frames.Frame:
        """
        Decode an incoming frame.

        """
        if frame.opcode in frames.CTRL_OPCODES:
            return frame
        if frame.opcode is frames.OP_CONT:
            if not self.decode_cont_data:
                return frame
            if frame.fin:
                self.decode_cont_data = False
        else:
            if not frame.rsv1:
                return frame
            frame = dataclasses.replace(frame, rsv1=False)
            if not frame.fin:
                self.decode_cont_data = True
            if self.remote_no_context_takeover:
                self.decoder = zlib.decompressobj(wbits=-self.remote_max_window_bits)
        data = frame.data
        if frame.fin:
            data += _EMPTY_UNCOMPRESSED_BLOCK
        max_length = 0 if max_size is None else max_size
        try:
            data = self.decoder.decompress(data, max_length)
        except zlib.error as exc:
            raise exceptions.ProtocolError('decompression failed') from exc
        if self.decoder.unconsumed_tail:
            raise exceptions.PayloadTooBig(f'over size limit (? > {max_size} bytes)')
        if frame.fin and self.remote_no_context_takeover:
            del self.decoder
        return dataclasses.replace(frame, data=data)

    def encode(self, frame: frames.Frame) -> frames.Frame:
        """
        Encode an outgoing frame.

        """
        if frame.opcode in frames.CTRL_OPCODES:
            return frame
        if frame.opcode is not frames.OP_CONT:
            frame = dataclasses.replace(frame, rsv1=True)
            if self.local_no_context_takeover:
                self.encoder = zlib.compressobj(wbits=-self.local_max_window_bits, **self.compress_settings)
        data = self.encoder.compress(frame.data) + self.encoder.flush(zlib.Z_SYNC_FLUSH)
        if frame.fin and data.endswith(_EMPTY_UNCOMPRESSED_BLOCK):
            data = data[:-4]
        if frame.fin and self.local_no_context_takeover:
            del self.encoder
        return dataclasses.replace(frame, data=data)