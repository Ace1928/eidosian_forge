import datetime
from google.auth import _helpers
from google.auth import credentials
from google.auth import crypt
from google.auth import exceptions
Checks if the credentials requires scopes.

        Returns:
            bool: True if there are no scopes set otherwise False.
        