import threading
from _thread import get_ident
from ... import branch as _mod_branch
from ... import debug, errors, osutils, registry, revision, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...lazy_import import lazy_import
from breezy.bzr import bzrdir
from breezy.bzr.bundle import serializer
import tempfile
class SmartServerRequest:
    """Base class for request handlers.

    To define a new request, subclass this class and override the `do` method
    (and if appropriate, `do_body` as well).  Request implementors should take
    care to call `translate_client_path` and `transport_from_client_path` as
    appropriate when dealing with paths received from the client.
    """

    def __init__(self, backing_transport, root_client_path='/', jail_root=None):
        """Constructor.

        :param backing_transport: the base transport to be used when performing
            this request.
        :param root_client_path: the client path that maps to the root of
            backing_transport.  This is used to interpret relpaths received
            from the client.  Clients will not be able to refer to paths above
            this root.  If root_client_path is None, then no translation will
            be performed on client paths.  Default is '/'.
        :param jail_root: if specified, the root of the BzrDir.open jail to use
            instead of backing_transport.
        """
        self._backing_transport = backing_transport
        if jail_root is None:
            jail_root = backing_transport
        self._jail_root = jail_root
        if root_client_path is not None:
            if not root_client_path.startswith('/'):
                root_client_path = '/' + root_client_path
            if not root_client_path.endswith('/'):
                root_client_path += '/'
        self._root_client_path = root_client_path
        self._body_chunks = []

    def _check_enabled(self):
        """Raises DisabledMethod if this method is disabled."""
        pass

    def do(self, *args):
        """Mandatory extension point for SmartServerRequest subclasses.

        Subclasses must implement this.

        This should return a SmartServerResponse if this command expects to
        receive no body.
        """
        raise NotImplementedError(self.do)

    def execute(self, *args):
        """Public entry point to execute this request.

        It will return a SmartServerResponse if the command does not expect a
        body.

        :param args: the arguments of the request.
        """
        self._check_enabled()
        return self.do(*args)

    def do_body(self, body_bytes):
        """Called if the client sends a body with the request.

        The do() method is still called, and must have returned None.

        Must return a SmartServerResponse.
        """
        if body_bytes != b'':
            raise errors.SmartProtocolError('Request does not expect a body')

    def do_chunk(self, chunk_bytes):
        """Called with each body chunk if the request has a streamed body.

        The do() method is still called, and must have returned None.
        """
        self._body_chunks.append(chunk_bytes)

    def do_end(self):
        """Called when the end of the request has been received."""
        body_bytes = b''.join(self._body_chunks)
        self._body_chunks = None
        return self.do_body(body_bytes)

    def setup_jail(self):
        jail_info.transports = [self._jail_root]

    def teardown_jail(self):
        jail_info.transports = None

    def translate_client_path(self, client_path):
        """Translate a path received from a network client into a local
        relpath.

        All paths received from the client *must* be translated.

        :param client_path: the path from the client.
        :returns: a relpath that may be used with self._backing_transport
            (unlike the untranslated client_path, which must not be used with
            the backing transport).
        """
        client_path = client_path.decode('utf-8')
        if self._root_client_path is None:
            return client_path
        if not client_path.startswith('/'):
            client_path = '/' + client_path
        if client_path + '/' == self._root_client_path:
            return '.'
        if client_path.startswith(self._root_client_path):
            path = client_path[len(self._root_client_path):]
            relpath = urlutils.joinpath('/', path)
            if not relpath.startswith('/'):
                raise ValueError(relpath)
            return urlutils.escape('.' + relpath)
        else:
            raise errors.PathNotChild(client_path, self._root_client_path)

    def transport_from_client_path(self, client_path):
        """Get a backing transport corresponding to the location referred to by
        a network client.

        :seealso: translate_client_path
        :returns: a transport cloned from self._backing_transport
        """
        relpath = self.translate_client_path(client_path)
        return self._backing_transport.clone(relpath)