import datetime
import json
import os
import socket
from oauth2client import _helpers
from oauth2client import client
class DevshellCredentials(client.GoogleCredentials):
    """Credentials object for Google Developer Shell environment.

    This object will allow a Google Developer Shell session to identify its
    user to Google and other OAuth 2.0 servers that can verify assertions. It
    can be used for the purpose of accessing data stored under the user
    account.

    This credential does not require a flow to instantiate because it
    represents a two legged flow, and therefore has all of the required
    information to generate and refresh its own access tokens.
    """

    def __init__(self, user_agent=None):
        super(DevshellCredentials, self).__init__(None, None, None, None, None, None, user_agent)
        self._refresh(None)

    def _refresh(self, http):
        """Refreshes the access token.

        Args:
            http: unused HTTP object
        """
        self.devshell_response = _SendRecv()
        self.access_token = self.devshell_response.access_token
        expires_in = self.devshell_response.expires_in
        if expires_in is not None:
            delta = datetime.timedelta(seconds=expires_in)
            self.token_expiry = client._UTCNOW() + delta
        else:
            self.token_expiry = None

    @property
    def user_email(self):
        return self.devshell_response.user_email

    @property
    def project_id(self):
        return self.devshell_response.project_id

    @classmethod
    def from_json(cls, json_data):
        raise NotImplementedError('Cannot load Developer Shell credentials from JSON.')

    @property
    def serialization_data(self):
        raise NotImplementedError('Cannot serialize Developer Shell credentials.')