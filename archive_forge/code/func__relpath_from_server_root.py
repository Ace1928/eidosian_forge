from .. import urlutils
from . import Server, Transport, register_transport, unregister_transport
def _relpath_from_server_root(self, relpath):
    unfiltered_path = urlutils.URL._combine_paths(self.base_path, relpath)
    if not unfiltered_path.startswith('/'):
        raise ValueError(unfiltered_path)
    return unfiltered_path[1:]