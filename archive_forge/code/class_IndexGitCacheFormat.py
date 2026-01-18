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
class IndexGitCacheFormat(BzrGitCacheFormat):

    def get_format_string(self):
        return b'bzr-git sha map with git object cache version 1\n'

    def initialize(self, transport):
        super().initialize(transport)
        transport.mkdir('index')
        transport.mkdir('objects')
        from .transportgit import TransportObjectStore
        TransportObjectStore.init(transport.clone('objects'))

    def open(self, transport):
        return IndexBzrGitCache(transport)