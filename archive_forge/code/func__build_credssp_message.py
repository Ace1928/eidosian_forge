import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _build_credssp_message(self, message, host):
    credssp_context = self.session.auth.contexts[host]
    sealed_message = credssp_context.wrap(message)
    cipher_negotiated = credssp_context.tls_connection.get_cipher_name()
    trailer_length = self._get_credssp_trailer_length(len(message), cipher_negotiated)
    return struct.pack('<i', trailer_length) + sealed_message