import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
class SessionSecurityTest(object):

    def wrap(self, message):
        encoded_message = base64.b64encode(message)
        signature = b'1234'
        return (encoded_message, signature)

    def unwrap(self, message, signature):
        assert signature == b'1234'
        decoded_message = base64.b64decode(message)
        return decoded_message