import logging
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _, _LI
class SwiftConnectionManager(object):
    """Connection Manager class responsible for initializing and managing
    swiftclient connections in store. The instance of that class can provide
    swift connections with a valid(and refreshed) user token if the token is
    going to expire soon.
    """
    AUTH_HEADER_NAME = 'X-Auth-Token'

    def __init__(self, store, store_location, context=None, allow_reauth=False):
        """Initialize manager with parameters required to establish connection.

        Initialize store and prepare it for interacting with swift. Also
        initialize keystone client that need to be used for authentication if
        allow_reauth is True.
        The method invariant is the following: if method was executed
        successfully and self.allow_reauth is True users can safely request
        valid(no expiration) swift connections any time. Otherwise, connection
        manager initialize a connection once and always returns that connection
        to users.

        :param store: store that provides connections
        :param store_location: image location in store
        :param context: user context to access data in Swift
        :param allow_reauth: defines if re-authentication need to be executed
        when a user request the connection
        """
        self._client = None
        self.store = store
        self.location = store_location
        self.context = context
        self.allow_reauth = allow_reauth
        self.storage_url = self._get_storage_url()
        self.connection = self._init_connection()

    def get_connection(self):
        """Get swift client connection.

        Returns swift client connection. If allow_reauth is True and
        connection token is going to expire soon then the method returns
        updated connection.
        The method invariant is the following: if self.allow_reauth is False
        then the method returns the same connection for every call. So the
        connection may expire. If self.allow_reauth is True the returned
        swift connection is always valid and cannot expire at least for
        swift_store_expire_soon_interval.
        """
        if self.allow_reauth:
            auth_ref = self.client.session.auth.auth_ref
            if self.store.backend_group:
                interval = getattr(self.store.conf, self.store.backend_group).swift_store_expire_soon_interval
            else:
                store_conf = self.store.conf.glance_store
                interval = store_conf.swift_store_expire_soon_interval
            if auth_ref.will_expire_soon(interval):
                LOG.info(_LI('Requesting new token for swift connection.'))
                auth_token = self.client.session.get_auth_headers().get(self.AUTH_HEADER_NAME)
                LOG.info(_LI('Token has been successfully requested. Refreshing swift connection.'))
                self.connection = self.store.get_store_connection(auth_token, self.storage_url)
        return self.connection

    @property
    def client(self):
        """Return keystone client to request a  new token.

        Initialize a client lazily from the method provided by glance_store.
        The method invariant is the following: if client cannot be
        initialized raise exception otherwise return initialized client that
        can be used for re-authentication any time.
        """
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_connection(self):
        """Initialize and return valid Swift connection."""
        auth_token = self.client.session.get_auth_headers().get(self.AUTH_HEADER_NAME)
        return self.store.get_store_connection(auth_token, self.storage_url)

    def _init_client(self):
        """Initialize Keystone client."""
        return self.store.init_client(location=self.location, context=self.context)

    def _get_storage_url(self):
        """Request swift storage url."""
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass