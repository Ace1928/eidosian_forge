from dulwich.object_store import tree_lookup_path
from .. import osutils
from ..bzr.versionedfile import UnavailableRepresentation
from ..errors import NoSuchRevision
from ..graph import Graph
from ..revision import NULL_REVISION
from .mapping import decode_git_path, encode_git_path
class GitBlobContentFactory:
    """Static data content factory.

    This takes a fulltext when created and just returns that during
    get_bytes_as('fulltext').

    :ivar sha1: None, or the sha1 of the content fulltext.
    :ivar storage_kind: The native storage kind of this factory. Always
        'fulltext'.
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: A tuple of parent keys for self.key. If the object has
        no parent information, None (as opposed to () for an empty list of
        parents).
     """

    def __init__(self, store, path, revision, blob_id):
        """Create a ContentFactory."""
        self.store = store
        self.key = (path, revision)
        self.storage_kind = 'git-blob'
        self.parents = None
        self.blob_id = blob_id
        self.size = None

    def get_bytes_as(self, storage_kind):
        if storage_kind == 'fulltext':
            return self.store[self.blob_id].as_raw_string()
        elif storage_kind == 'lines':
            return list(osutils.chunks_to_lines(self.store[self.blob_id].as_raw_chunks()))
        elif storage_kind == 'chunked':
            return self.store[self.blob_id].as_raw_chunks()
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        if storage_kind == 'lines':
            return iter(osutils.chunks_to_lines(self.store[self.blob_id].as_raw_chunks()))
        elif storage_kind == 'chunked':
            return iter(self.store[self.blob_id].as_raw_chunks())
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)