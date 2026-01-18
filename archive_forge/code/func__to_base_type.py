import logging
from google.appengine.ext import ndb
from oauth2client import client
def _to_base_type(self, value):
    """Converts our validated value to a JSON serialized string.

        Args:
            value: A value to be set in the datastore.

        Returns:
            A JSON serialized version of the credential, else '' if value
            is None.
        """
    if value is None:
        return ''
    else:
        return value.to_json()