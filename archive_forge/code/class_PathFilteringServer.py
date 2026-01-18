from .. import urlutils
from . import Server, Transport, register_transport, unregister_transport
class PathFilteringServer(Server):
    """Transport server for PathFilteringTransport.

    It holds the backing_transport and filter_func for PathFilteringTransports.
    All paths will be passed through filter_func before calling into the
    backing_transport.

    Note that paths returned from the backing transport are *not* altered in
    anyway.  So, depending on the filter_func, PathFilteringTransports might
    not conform to the usual expectations of Transport behaviour; e.g. 'name'
    in t.list_dir('dir') might not imply t.has('dir/name') is True!  A filter
    that merely prefixes a constant path segment will be essentially
    transparent, whereas a filter that does rot13 to paths will break
    expectations and probably cause confusing errors.  So choose your
    filter_func with care.
    """

    def __init__(self, backing_transport, filter_func):
        """Constructor.

        :param backing_transport: a transport
        :param filter_func: a callable that takes paths, and translates them
            into paths for use with the backing transport.
        """
        self.backing_transport = backing_transport
        self.filter_func = filter_func

    def _factory(self, url):
        return PathFilteringTransport(self, url)

    def get_url(self):
        return self.scheme

    def start_server(self):
        self.scheme = 'filtered-%d:///' % id(self)
        register_transport(self.scheme, self._factory)

    def stop_server(self):
        unregister_transport(self.scheme, self._factory)