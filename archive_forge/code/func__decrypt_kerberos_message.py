import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _decrypt_kerberos_message(self, encrypted_data, host):
    signature_length = struct.unpack('<i', encrypted_data[:4])[0]
    signature = encrypted_data[4:signature_length + 4]
    encrypted_message = encrypted_data[signature_length + 4:]
    message = self.session.auth.unwrap_winrm(host, encrypted_message, signature)
    return message