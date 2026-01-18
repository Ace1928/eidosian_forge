import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import baseauthenticator
def _BuildAuthenticatorResponse(self, app_id, client_data, plugin_response):
    """Builds the response to return to the caller."""
    encoded_client_data = self._Base64Encode(client_data)
    signature_data = str(plugin_response['signatureData'])
    key_handle = str(plugin_response['keyHandle'])
    response = {'clientData': encoded_client_data, 'signatureData': signature_data, 'applicationId': app_id, 'keyHandle': key_handle}
    return response