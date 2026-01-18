import datetime
import json
import os
import socket
from oauth2client import _helpers
from oauth2client import client
class CredentialInfoResponse(object):
    """Credential information response from Developer Shell server.

    The credential information response from Developer Shell socket is a
    PBLite-formatted JSON array with fields encoded by their index in the
    array:

    * Index 0 - user email
    * Index 1 - default project ID. None if the project context is not known.
    * Index 2 - OAuth2 access token. None if there is no valid auth context.
    * Index 3 - Seconds until the access token expires. None if not present.
    """

    def __init__(self, json_string):
        """Initialize the response data from JSON PBLite array."""
        pbl = json.loads(json_string)
        if not isinstance(pbl, list):
            raise ValueError('Not a list: ' + str(pbl))
        pbl_len = len(pbl)
        self.user_email = pbl[0] if pbl_len > 0 else None
        self.project_id = pbl[1] if pbl_len > 1 else None
        self.access_token = pbl[2] if pbl_len > 2 else None
        self.expires_in = pbl[3] if pbl_len > 3 else None