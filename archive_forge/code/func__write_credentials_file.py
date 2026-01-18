import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _write_credentials_file(credentials_file, credentials):
    """Writes credentials to a file.

    Refer to :func:`_load_credentials_file` for the format.

    Args:
        credentials_file: An open file handle, must be read/write.
        credentials: A dictionary mapping user-defined keys to an instance of
            :class:`oauth2client.client.Credentials`.
    """
    data = {'file_version': 2, 'credentials': {}}
    for key, credential in iteritems(credentials):
        credential_json = credential.to_json()
        encoded_credential = _helpers._from_bytes(base64.b64encode(_helpers._to_bytes(credential_json)))
        data['credentials'][key] = encoded_credential
    credentials_file.seek(0)
    json.dump(data, credentials_file)
    credentials_file.truncate()