import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def _decrypt_credssp_message(self, encrypted_data, host):
    encrypted_message = encrypted_data[4:]
    credssp_context = self.session.auth.contexts[host]
    message = credssp_context.unwrap(encrypted_message)
    return message