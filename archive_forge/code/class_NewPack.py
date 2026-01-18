import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
class NewPack(Pack):
    """An in memory proxy for a pack which is being created."""

    def __init__(self, pack_collection, upload_suffix='', file_mode=None):
        """Create a NewPack instance.

        :param pack_collection: A PackCollection into which this is being inserted.
        :param upload_suffix: An optional suffix to be given to any temporary
            files created during the pack creation. e.g '.autopack'
        :param file_mode: Unix permissions for newly created file.
        """
        index_builder_class = pack_collection._index_builder_class
        if pack_collection.chk_index is not None:
            chk_index = index_builder_class(reference_lists=0)
        else:
            chk_index = None
        Pack.__init__(self, index_builder_class(reference_lists=1), index_builder_class(reference_lists=2), index_builder_class(reference_lists=2, key_elements=2), index_builder_class(reference_lists=0), chk_index=chk_index)
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
            mutter('%s: create_pack: pack stream open: %s%s t+%6.3fs', time.ctime(), self.upload_transport.base, self.random_name, time.time() - self.start_time)
        self._buffer = [[], 0]

        def _write_data(bytes, flush=False, _buffer=self._buffer, _write=self.write_stream.write, _update=self._hash.update):
            _buffer[0].append(bytes)
            _buffer[1] += len(bytes)
            if _buffer[1] > self._cache_limit or flush:
                bytes = b''.join(_buffer[0])
                _write(bytes)
                _update(bytes)
                _buffer[:] = [[], 0]
        self._write_data = _write_data
        self._writer = pack.ContainerWriter(self._write_data)
        self._writer.begin()
        self._state = 'open'
        self.name = None

    def abort(self):
        """Cancel creating this pack."""
        self._state = 'aborted'
        self.write_stream.close()
        self.upload_transport.delete(self.random_name)

    def access_tuple(self):
        """Return a tuple (transport, name) for the pack content."""
        if self._state == 'finished':
            return Pack.access_tuple(self)
        elif self._state == 'open':
            return (self.upload_transport, self.random_name)
        else:
            raise AssertionError(self._state)

    def data_inserted(self):
        """True if data has been added to this pack."""
        return bool(self.get_revision_count() or self.inventory_index.key_count() or self.text_index.key_count() or self.signature_index.key_count() or (self.chk_index is not None and self.chk_index.key_count()))

    def finish_content(self):
        if self.name is not None:
            return
        self._writer.end()
        if self._buffer[1]:
            self._write_data(b'', flush=True)
        self.name = self._hash.hexdigest()

    def finish(self, suspend=False):
        """Finish the new pack.

        This:
         - finalises the content
         - assigns a name (the md5 of the content, currently)
         - writes out the associated indices
         - renames the pack into place.
         - stores the index size tuple for the pack in the index_sizes
           attribute.
        """
        self.finish_content()
        if not suspend:
            self._check_references()
        self.index_sizes = [None, None, None, None]
        self._write_index('revision', self.revision_index, 'revision', suspend)
        self._write_index('inventory', self.inventory_index, 'inventory', suspend)
        self._write_index('text', self.text_index, 'file texts', suspend)
        self._write_index('signature', self.signature_index, 'revision signatures', suspend)
        if self.chk_index is not None:
            self.index_sizes.append(None)
            self._write_index('chk', self.chk_index, 'content hash bytes', suspend)
        self.write_stream.close(want_fdatasync=self._pack_collection.config_stack.get('repository.fdatasync'))
        new_name = self.name + '.pack'
        if not suspend:
            new_name = '../packs/' + new_name
        self.upload_transport.move(self.random_name, new_name)
        self._state = 'finished'
        if 'pack' in debug.debug_flags:
            mutter('%s: create_pack: pack finished: %s%s->%s t+%6.3fs', time.ctime(), self.upload_transport.base, self.random_name, new_name, time.time() - self.start_time)

    def flush(self):
        """Flush any current data."""
        if self._buffer[1]:
            bytes = b''.join(self._buffer[0])
            self.write_stream.write(bytes)
            self._hash.update(bytes)
            self._buffer[:] = [[], 0]

    def _get_external_refs(self, index):
        return index._external_references()

    def set_write_cache_size(self, size):
        self._cache_limit = size

    def _write_index(self, index_type, index, label, suspend=False):
        """Write out an index.

        :param index_type: The type of index to write - e.g. 'revision'.
        :param index: The index object to serialise.
        :param label: What label to give the index e.g. 'revision'.
        """
        index_name = self.index_name(index_type, self.name)
        if suspend:
            transport = self.upload_transport
        else:
            transport = self.index_transport
        index_tempfile = index.finish()
        index_bytes = index_tempfile.read()
        write_stream = transport.open_write_stream(index_name, mode=self._file_mode)
        write_stream.write(index_bytes)
        write_stream.close(want_fdatasync=self._pack_collection.config_stack.get('repository.fdatasync'))
        self.index_sizes[self.index_offset(index_type)] = len(index_bytes)
        if 'pack' in debug.debug_flags:
            mutter('%s: create_pack: wrote %s index: %s%s t+%6.3fs', time.ctime(), label, self.upload_transport.base, self.random_name, time.time() - self.start_time)
        self._replace_index_with_readonly(index_type)