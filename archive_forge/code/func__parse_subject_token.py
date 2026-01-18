import json
import os
import subprocess
import sys
import time
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _parse_subject_token(self, response):
    self._validate_response_schema(response)
    if not response['success']:
        if 'code' not in response or 'message' not in response:
            raise exceptions.MalformedError('Error code and message fields are required in the response.')
        raise exceptions.RefreshError('Executable returned unsuccessful response: code: {}, message: {}.'.format(response['code'], response['message']))
    if 'expiration_time' in response and response['expiration_time'] < time.time():
        raise exceptions.RefreshError('The token returned by the executable is expired.')
    if 'token_type' not in response:
        raise exceptions.MalformedError('The executable response is missing the token_type field.')
    if response['token_type'] == 'urn:ietf:params:oauth:token-type:jwt' or response['token_type'] == 'urn:ietf:params:oauth:token-type:id_token':
        return response['id_token']
    elif response['token_type'] == 'urn:ietf:params:oauth:token-type:saml2':
        return response['saml_response']
    else:
        raise exceptions.RefreshError('Executable returned unsupported token type.')