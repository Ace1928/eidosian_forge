import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
class GCPack(NewPack):

    def __init__(self, pack_collection, upload_suffix='', file_mode=None):
        """Create a NewPack instance.

        :param pack_collection: A PackCollection into which this is being
            inserted.
        :param upload_suffix: An optional suffix to be given to any temporary
            files created during the pack creation. e.g '.autopack'
        :param file_mode: An optional file mode to create the new files with.
        """
        index_builder_class = pack_collection._index_builder_class
        if pack_collection.chk_index is not None:
            chk_index = index_builder_class(reference_lists=0)
        else:
            chk_index = None
        Pack.__init__(self, index_builder_class(reference_lists=1), index_builder_class(reference_lists=1), index_builder_class(reference_lists=1, key_elements=2), index_builder_class(reference_lists=0), chk_index=chk_index)
        self._pack_collection = pack_collection
        self.index_class = pack_collection._index_class
        self.upload_transport = pack_collection._upload_transport
        self.index_transport = pack_collection._index_transport
        self.pack_transport = pack_collection._pack_transport
        self._file_mode = file_mode
        self._hash = osutils.md5()
        self.index_sizes = None
        self._cache_limit = 0
        self.random_name = osutils.rand_chars(20) + upload_suffix
        self.start_time = time.time()
        self.write_stream = self.upload_transport.open_write_stream(self.random_name, mode=self._file_mode)
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: pack stream open: %s%s t+%6.3fs', time.ctime(), self.upload_transport.base, self.random_name, time.time() - self.start_time)
        self._buffer = [[], 0]

        def _write_data(data, flush=False, _buffer=self._buffer, _write=self.write_stream.write, _update=self._hash.update):
            _buffer[0].append(data)
            _buffer[1] += len(data)
            if _buffer[1] > self._cache_limit or flush:
                data = b''.join(_buffer[0])
                _write(data)
                _update(data)
                _buffer[:] = [[], 0]
        self._write_data = _write_data
        self._writer = pack.ContainerWriter(self._write_data)
        self._writer.begin()
        self._state = 'open'
        self.name = None

    def _check_references(self):
        """Make sure our external references are present.

        Packs are allowed to have deltas whose base is not in the pack, but it
        must be present somewhere in this collection.  It is not allowed to
        have deltas based on a fallback repository.
        (See <https://bugs.launchpad.net/bzr/+bug/288751>)
        """