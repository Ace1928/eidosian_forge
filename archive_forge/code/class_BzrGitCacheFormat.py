import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
class BzrGitCacheFormat:
    """Bazaar-Git Cache Format."""

    def get_format_string(self):
        """Return a single-line unique format string for this cache format."""
        raise NotImplementedError(self.get_format_string)

    def open(self, transport):
        """Open this format on a transport."""
        raise NotImplementedError(self.open)

    def initialize(self, transport):
        """Create a new instance of this cache format at transport."""
        transport.put_bytes('format', self.get_format_string())

    @classmethod
    def from_transport(self, transport):
        """Open a cache file present on a transport, or initialize one.

        :param transport: Transport to use
        :return: A BzrGitCache instance
        """
        try:
            format_name = transport.get_bytes('format')
            format = formats.get(format_name)
        except NoSuchFile:
            format = formats.get('default')
            format.initialize(transport)
        return format.open(transport)

    @classmethod
    def from_repository(cls, repository):
        """Open a cache file for a repository.

        This will use the repository's transport to store the cache file, or
        use the users global cache directory if the repository has no
        transport associated with it.

        :param repository: Repository to open the cache for
        :return: A `BzrGitCache`
        """
        from ..transport.local import LocalTransport
        repo_transport = getattr(repository, '_transport', None)
        if repo_transport is not None and isinstance(repo_transport, LocalTransport):
            try:
                repo_transport = remove_readonly_transport_decorator(repo_transport)
            except bzr_errors.ReadOnlyError:
                transport = None
            else:
                try:
                    repo_transport.mkdir('git')
                except FileExists:
                    pass
                transport = repo_transport.clone('git')
        else:
            transport = None
        if transport is None:
            transport = get_remote_cache_transport(repository)
        return cls.from_transport(transport)