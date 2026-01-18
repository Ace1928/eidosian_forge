import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _load_credentials_file(credentials_file):
    """Load credentials from the given file handle.

    The file is expected to be in this format:

        {
            "file_version": 2,
            "credentials": {
                "key": "base64 encoded json representation of credentials."
            }
        }

    This function will warn and return empty credentials instead of raising
    exceptions.

    Args:
        credentials_file: An open file handle.

    Returns:
        A dictionary mapping user-defined keys to an instance of
        :class:`oauth2client.client.Credentials`.
    """
    try:
        credentials_file.seek(0)
        data = json.load(credentials_file)
    except Exception:
        logger.warning('Credentials file could not be loaded, will ignore and overwrite.')
        return {}
    if data.get('file_version') != 2:
        logger.warning('Credentials file is not version 2, will ignore and overwrite.')
        return {}
    credentials = {}
    for key, encoded_credential in iteritems(data.get('credentials', {})):
        try:
            credential_json = base64.b64decode(encoded_credential)
            credential = client.Credentials.new_from_json(credential_json)
            credentials[key] = credential
        except:
            logger.warning('Invalid credential {0} in file, ignoring.'.format(key))
    return credentials