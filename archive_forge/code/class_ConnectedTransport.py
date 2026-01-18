import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class ConnectedTransport(Transport):
    """A transport connected to a remote server.

    This class provide the basis to implement transports that need to connect
    to a remote server.

    Host and credentials are available as private attributes, cloning preserves
    them and share the underlying, protocol specific, connection.
    """

    def __init__(self, base, _from_transport=None):
        """Constructor.

        The caller should ensure that _from_transport points at the same host
        as the new base.

        :param base: transport root URL

        :param _from_transport: optional transport to build from. The built
            transport will share the connection with this transport.
        """
        if not base.endswith('/'):
            base += '/'
        self._parsed_url = self._split_url(base)
        if _from_transport is not None:
            self._parsed_url.password = _from_transport._parsed_url.password
            self._parsed_url.quoted_password = _from_transport._parsed_url.quoted_password
        base = str(self._parsed_url)
        super().__init__(base)
        if _from_transport is None:
            self._shared_connection = _SharedConnection()
        else:
            self._shared_connection = _from_transport._shared_connection

    @property
    def _user(self):
        return self._parsed_url.user

    @property
    def _password(self):
        return self._parsed_url.password

    @property
    def _host(self):
        return self._parsed_url.host

    @property
    def _port(self):
        return self._parsed_url.port

    @property
    def _path(self):
        return self._parsed_url.path

    @property
    def _scheme(self):
        return self._parsed_url.scheme

    def clone(self, offset=None):
        """Return a new transport with root at self.base + offset

        We leave the daughter classes take advantage of the hint
        that it's a cloning not a raw creation.
        """
        if offset is None:
            return self.__class__(self.base, _from_transport=self)
        else:
            return self.__class__(self.abspath(offset), _from_transport=self)

    @staticmethod
    def _split_url(url):
        return urlutils.URL.from_string(url)

    @staticmethod
    def _unsplit_url(scheme, user, password, host, port, path):
        """Build the full URL for the given already URL encoded path.

        user, password, host and path will be quoted if they contain reserved
        chars.

        Args:
          scheme: protocol
          user: login
          password: associated password
          host: the server address
          port: the associated port
          path: the absolute path on the server

        :return: The corresponding URL.
        """
        netloc = urlutils.quote(host)
        if user is not None:
            netloc = '{}@{}'.format(urlutils.quote(user), netloc)
        if port is not None:
            netloc = '%s:%d' % (netloc, port)
        path = urlutils.escape(path)
        return urlutils.urlparse.urlunparse((scheme, netloc, path, None, None, None))

    def relpath(self, abspath):
        """Return the local path portion from a given absolute path"""
        parsed_url = self._split_url(abspath)
        error = []
        if parsed_url.scheme != self._parsed_url.scheme:
            error.append('scheme mismatch')
        if parsed_url.user != self._parsed_url.user:
            error.append('user name mismatch')
        if parsed_url.host != self._parsed_url.host:
            error.append('host mismatch')
        if parsed_url.port != self._parsed_url.port:
            error.append('port mismatch')
        if not (parsed_url.path == self._parsed_url.path[:-1] or parsed_url.path.startswith(self._parsed_url.path)):
            error.append('path mismatch')
        if error:
            extra = ', '.join(error)
            raise errors.PathNotChild(abspath, self.base, extra=extra)
        pl = len(self._parsed_url.path)
        return parsed_url.path[pl:].strip('/')

    def abspath(self, relpath):
        """Return the full url to the given relative path.

        Args:
          relpath: the relative path urlencoded

        :returns: the Unicode version of the absolute path for relpath.
        """
        return str(self._parsed_url.clone(relpath))

    def _remote_path(self, relpath):
        """Return the absolute path part of the url to the given relative path.

        This is the path that the remote server expect to receive in the
        requests, daughter classes should redefine this method if needed and
        use the result to build their requests.

        Args:
          relpath: the path relative to the transport base urlencoded.

        :return: the absolute Unicode path on the server,
        """
        return self._parsed_url.clone(relpath).path

    def _get_shared_connection(self):
        """Get the object shared amongst cloned transports.

        This should be used only by classes that needs to extend the sharing
        with objects other than transports.

        Use _get_connection to get the connection itself.
        """
        return self._shared_connection

    def _set_connection(self, connection, credentials=None):
        """Record a newly created connection with its associated credentials.

        Note: To ensure that connection is still shared after a temporary
        failure and a new one needs to be created, daughter classes should
        always call this method to set the connection and do so each time a new
        connection is created.

        Args:
          connection: An opaque object representing the connection used by
            the daughter class.
          credentials: An opaque object representing the credentials
            needed to create the connection.
        """
        self._shared_connection.connection = connection
        self._shared_connection.credentials = credentials
        for hook in self.hooks['post_connect']:
            hook(self)

    def _get_connection(self):
        """Returns the transport specific connection object."""
        return self._shared_connection.connection

    def _get_credentials(self):
        """Returns the credentials used to establish the connection."""
        return self._shared_connection.credentials

    def _update_credentials(self, credentials):
        """Update the credentials of the current connection.

        Some protocols can renegociate the credentials within a connection,
        this method allows daughter classes to share updated credentials.

        :param credentials: the updated credentials.
        """
        self._shared_connection.credentials = credentials

    def _reuse_for(self, other_base):
        """Returns a transport sharing the same connection if possible.

        Note: we share the connection if the expected credentials are the
        same: (host, port, user). Some protocols may disagree and redefine the
        criteria in daughter classes.

        Note: we don't compare the passwords here because other_base may have
        been obtained from an existing transport.base which do not mention the
        password.

        :param other_base: the URL we want to share the connection with.

        :return: A new transport or None if the connection cannot be shared.
        """
        try:
            parsed_url = self._split_url(other_base)
        except urlutils.InvalidURL:
            return None
        transport = None
        if parsed_url.scheme == self._parsed_url.scheme and parsed_url.user == self._parsed_url.user and (parsed_url.host == self._parsed_url.host) and (parsed_url.port == self._parsed_url.port):
            path = parsed_url.path
            if not path.endswith('/'):
                path += '/'
            if self._parsed_url.path == path:
                return self
            transport = self.__class__(other_base, _from_transport=self)
        return transport

    def disconnect(self):
        """Disconnect the transport.

        If and when required the transport willl reconnect automatically.
        """
        raise NotImplementedError(self.disconnect)