import abc
import os
import six
from google.auth import _helpers, environment_vars
from google.auth import exceptions
class CredentialsWithTokenUri(Credentials):
    """Abstract base for credentials supporting ``with_token_uri`` factory"""

    def with_token_uri(self, token_uri):
        """Returns a copy of these credentials with a modified token uri.

        Args:
            token_uri (str): The uri to use for fetching/exchanging tokens

        Returns:
            google.oauth2.credentials.Credentials: A new credentials instance.
        """
        raise NotImplementedError('This credential does not use token uri.')