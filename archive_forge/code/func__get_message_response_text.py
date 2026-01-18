from __future__ import unicode_literals
import sys
import os
import requests
import requests.auth
import warnings
from winrm.exceptions import InvalidCredentialsError, WinRMError, WinRMTransportError
from winrm.encryption import Encryption
def _get_message_response_text(self, response):
    if self.encryption:
        response_text = self.encryption.parse_encrypted_response(response)
    else:
        response_text = response.content
    return response_text