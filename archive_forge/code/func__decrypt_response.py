import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _decrypt_response(self, response, host):
    parts = response.content.split(self.MIME_BOUNDARY + b'\r\n')
    parts = list(filter(None, parts))
    message = b''
    for i in range(0, len(parts)):
        if i % 2 == 1:
            continue
        header = parts[i].strip()
        payload = parts[i + 1]
        expected_length = int(header.split(b'Length=')[1])
        if payload.endswith(self.MIME_BOUNDARY + b'--\r\n'):
            payload = payload[:len(payload) - 24]
        encrypted_data = payload.replace(b'\tContent-Type: application/octet-stream\r\n', b'')
        decrypted_message = self._decrypt_message(encrypted_data, host)
        actual_length = len(decrypted_message)
        if actual_length != expected_length:
            raise WinRMError('Encrypted length from server does not match the expected size, message has been tampered with')
        message += decrypted_message
    return message