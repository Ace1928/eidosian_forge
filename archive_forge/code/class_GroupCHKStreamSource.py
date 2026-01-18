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
class GroupCHKStreamSource(StreamSource):
    """Used when both the source and target repo are GroupCHK repos."""

    def __init__(self, from_repository, to_format):
        """Create a StreamSource streaming from from_repository."""
        super().__init__(from_repository, to_format)
        self._revision_keys = None
        self._text_keys = None
        self._text_fetch_order = 'groupcompress'
        self._chk_id_roots = None
        self._chk_p_id_roots = None

    def _get_inventory_stream(self, inventory_keys, allow_absent=False):
        """Get a stream of inventory texts.

        When this function returns, self._chk_id_roots and self._chk_p_id_roots
        should be populated.
        """
        self._chk_id_roots = []
        self._chk_p_id_roots = []

        def _filtered_inv_stream():
            id_roots_set = set()
            p_id_roots_set = set()
            source_vf = self.from_repository.inventories
            stream = source_vf.get_record_stream(inventory_keys, 'groupcompress', True)
            for record in stream:
                if record.storage_kind == 'absent':
                    if allow_absent:
                        continue
                    else:
                        raise errors.NoSuchRevision(self, record.key)
                lines = record.get_bytes_as('lines')
                chk_inv = inventory.CHKInventory.deserialise(None, lines, record.key)
                key = chk_inv.id_to_entry.key()
                if key not in id_roots_set:
                    self._chk_id_roots.append(key)
                    id_roots_set.add(key)
                p_id_map = chk_inv.parent_id_basename_to_file_id
                if p_id_map is None:
                    raise AssertionError('Parent id -> file_id map not set')
                key = p_id_map.key()
                if key not in p_id_roots_set:
                    p_id_roots_set.add(key)
                    self._chk_p_id_roots.append(key)
                yield record
            id_roots_set.clear()
            p_id_roots_set.clear()
        return ('inventories', _filtered_inv_stream())

    def _get_filtered_chk_streams(self, excluded_revision_keys):
        self._text_keys = set()
        excluded_revision_keys.discard(_mod_revision.NULL_REVISION)
        if not excluded_revision_keys:
            uninteresting_root_keys = set()
            uninteresting_pid_root_keys = set()
        else:
            present_keys = self.from_repository._find_present_inventory_keys(excluded_revision_keys)
            present_ids = [k[-1] for k in present_keys]
            uninteresting_root_keys = set()
            uninteresting_pid_root_keys = set()
            for inv in self.from_repository.iter_inventories(present_ids):
                uninteresting_root_keys.add(inv.id_to_entry.key())
                uninteresting_pid_root_keys.add(inv.parent_id_basename_to_file_id.key())
        chk_bytes = self.from_repository.chk_bytes

        def _filter_id_to_entry():
            interesting_nodes = chk_map.iter_interesting_nodes(chk_bytes, self._chk_id_roots, uninteresting_root_keys)
            for record in _filter_text_keys(interesting_nodes, self._text_keys, chk_map._bytes_to_text_key):
                if record is not None:
                    yield record
            self._chk_id_roots = None
        yield ('chk_bytes', _filter_id_to_entry())

        def _get_parent_id_basename_to_file_id_pages():
            for record, items in chk_map.iter_interesting_nodes(chk_bytes, self._chk_p_id_roots, uninteresting_pid_root_keys):
                if record is not None:
                    yield record
            self._chk_p_id_roots = None
        yield ('chk_bytes', _get_parent_id_basename_to_file_id_pages())

    def _get_text_stream(self):
        text_stream = self.from_repository.texts.get_record_stream(self._text_keys, self._text_fetch_order, False)
        return ('texts', text_stream)

    def get_stream(self, search):

        def wrap_and_count(pb, rc, stream):
            """Yield records from stream while showing progress."""
            count = 0
            for record in stream:
                if count == rc.STEP:
                    rc.increment(count)
                    pb.update('Estimate', rc.current, rc.max)
                    count = 0
                count += 1
                yield record
        revision_ids = search.get_keys()
        with ui.ui_factory.nested_progress_bar() as pb:
            rc = self._record_counter
            self._record_counter.setup(len(revision_ids))
            for stream_info in self._fetch_revision_texts(revision_ids):
                yield (stream_info[0], wrap_and_count(pb, rc, stream_info[1]))
            self._revision_keys = [(rev_id,) for rev_id in revision_ids]
            from_repo = self.from_repository
            parent_keys = from_repo._find_parent_keys_of_revisions(self._revision_keys)
            self.from_repository.revisions.clear_cache()
            self.from_repository.signatures.clear_cache()
            self.from_repository._unstacked_provider.disable_cache()
            self.from_repository._unstacked_provider.enable_cache()
            s = self._get_inventory_stream(self._revision_keys)
            yield (s[0], wrap_and_count(pb, rc, s[1]))
            self.from_repository.inventories.clear_cache()
            for stream_info in self._get_filtered_chk_streams(parent_keys):
                yield (stream_info[0], wrap_and_count(pb, rc, stream_info[1]))
            self.from_repository.chk_bytes.clear_cache()
            s = self._get_text_stream()
            yield (s[0], wrap_and_count(pb, rc, s[1]))
            self.from_repository.texts.clear_cache()
            pb.update('Done', rc.max, rc.max)

    def get_stream_for_missing_keys(self, missing_keys):
        missing_inventory_keys = set()
        for key in missing_keys:
            if key[0] != 'inventories':
                raise AssertionError('The only missing keys we should be filling in are inventory keys, not %s' % (key[0],))
            missing_inventory_keys.add(key[1:])
        if self._chk_id_roots or self._chk_p_id_roots:
            raise AssertionError('Cannot call get_stream_for_missing_keys until all of get_stream() has been consumed.')
        yield self._get_inventory_stream(missing_inventory_keys, allow_absent=True)
        yield from self._get_filtered_chk_streams(set())