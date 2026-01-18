import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _decode_credential_from_json(self, cred_entry):
    """Load a credential from our JSON serialization.

        Args:
            cred_entry: A dict entry from the data member of our format

        Returns:
            (key, cred) where the key is the key tuple and the cred is the
            OAuth2Credential object.
        """
    raw_key = cred_entry['key']
    key = _dict_to_tuple_key(raw_key)
    credential = None
    credential = client.Credentials.new_from_json(json.dumps(cred_entry['credential']))
    return (key, credential)