import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
class CredSSPContext(object):

    def __init__(self):

        class TlsContext(object):

            def get_cipher_name(self):
                return 'ECDH-RSA-AES256-SHA'
        self.tls_connection = TlsContext()
        self.session_security = SessionSecurityTest()

    def wrap(self, message):
        encoded_message, signature = self.session_security.wrap(message)
        return encoded_message

    def unwrap(self, message):
        decoded_mesage = self.session_security.unwrap(message, b'1234')
        return decoded_mesage