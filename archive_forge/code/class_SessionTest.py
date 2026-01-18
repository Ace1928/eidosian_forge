import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
class SessionTest(object):

    def __init__(self):
        self.auth = AuthTest()

    def prepare_request(self, request):
        request.body = request.data
        return request