import ssl
import struct
import typing as t
from ._certificate import get_tls_server_end_point_data
from .client import Credential, MessageEncryptor, SyncLDAPClient
class SpnegoEncryptor(MessageEncryptor):

    def __init__(self, context: 'spnego.ContextProxy') -> None:
        self.context = context

    def wrap(self, data: bytes) -> bytes:
        wrapped_data = self.context.wrap(data, encrypt=True).data
        return len(wrapped_data).to_bytes(4, byteorder='big') + wrapped_data

    def unwrap(self, data: bytes) -> t.Tuple[bytes, int]:
        data_view = memoryview(data)
        data_len = struct.unpack('>I', data_view[:4])[0]
        data_view = data_view[4:]
        if len(data_view) < data_len:
            return (b'', 0)
        data_view = data_view[:data_len]
        return (self.context.unwrap(data_view.tobytes()).data, data_len + 4)