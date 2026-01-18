import datetime
import json
import time
from urllib.parse import urljoin
from keystoneauth1 import discover
from keystoneauth1 import plugin
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.identity import base
class AccessInfoV1:
    """An object for encapsulating a raw v1 auth token."""

    def __init__(self, auth_url, storage_url, account, username, auth_token, token_life):
        self.auth_url = auth_url
        self.storage_url = storage_url
        self.account = account
        self.service_catalog = ServiceCatalogV1(auth_url, storage_url, account)
        self.username = username
        self.auth_token = auth_token
        self._issued = time.time()
        try:
            self._expires = self._issued + float(token_life)
        except (TypeError, ValueError):
            self._expires = None
        self.project_id = None

    @property
    def expires(self):
        if self._expires is None:
            return None
        return datetime.datetime.fromtimestamp(self._expires, UTC)

    @property
    def issued(self):
        return datetime.datetime.fromtimestamp(self._issued, UTC)

    @property
    def user_id(self):
        return self.username

    def will_expire_soon(self, stale_duration):
        """Determines if expiration is about to occur.

        :returns: true if expiration is within the given duration
        """
        if self._expires is None:
            return False
        return time.time() + stale_duration > self._expires

    def get_state(self):
        """Serialize the current state."""
        return json.dumps({'auth_url': self.auth_url, 'storage_url': self.storage_url, 'account': self.account, 'username': self.username, 'auth_token': self.auth_token, 'issued': self._issued, 'expires': self._expires}, sort_keys=True)

    @classmethod
    def from_state(cls, data):
        """Deserialize the given state.

        :returns: a new AccessInfoV1 object with the given state
        """
        data = json.loads(data)
        access = cls(data['auth_url'], data['storage_url'], data['account'], data['username'], data['auth_token'], token_life=None)
        access._issued = data['issued']
        access._expires = data['expires']
        return access