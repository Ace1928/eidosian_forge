from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
class StreamSource:
    """A source of a stream for fetching between repositories."""

    def __init__(self, from_repository, to_format):
        """Create a StreamSource streaming from from_repository."""
        self.from_repository = from_repository
        self.to_format = to_format
        from .recordcounter import RecordCounter
        self._record_counter = RecordCounter()

    def delta_on_metadata(self):
        """Return True if delta's are permitted on metadata streams.

        That is on revisions and signatures.
        """
        src_serializer = self.from_repository._format._serializer
        target_serializer = self.to_format._serializer
        return self.to_format._fetch_uses_deltas and src_serializer == target_serializer

    def _fetch_revision_texts(self, revs):
        from_sf = self.from_repository.signatures
        keys = [(rev_id,) for rev_id in revs]
        signatures = versionedfile.filter_absent(from_sf.get_record_stream(keys, self.to_format._fetch_order, not self.to_format._fetch_uses_deltas))
        from_rf = self.from_repository.revisions
        revisions = from_rf.get_record_stream(keys, self.to_format._fetch_order, not self.delta_on_metadata())
        return [('signatures', signatures), ('revisions', revisions)]

    def _generate_root_texts(self, revs):
        """This will be called by get_stream between fetching weave texts and
        fetching the inventory weave.
        """
        if self._rich_root_upgrade():
            return _mod_fetch.Inter1and2Helper(self.from_repository).generate_root_texts(revs)
        else:
            return []

    def get_stream(self, search):
        phase = 'file'
        revs = search.get_keys()
        graph = self.from_repository.get_graph()
        revs = tsort.topo_sort(graph.get_parent_map(revs))
        data_to_fetch = self.from_repository.item_keys_introduced_by(revs)
        text_keys = []
        for knit_kind, file_id, revisions in data_to_fetch:
            if knit_kind != phase:
                phase = knit_kind
            if knit_kind == 'file':
                text_keys.extend([(file_id, revision) for revision in revisions])
            elif knit_kind == 'inventory':
                from_texts = self.from_repository.texts
                yield ('texts', from_texts.get_record_stream(text_keys, self.to_format._fetch_order, not self.to_format._fetch_uses_deltas))
                text_keys = None
                yield from self._generate_root_texts(revs)
                yield from self._get_inventory_stream(revs)
            elif knit_kind == 'signatures':
                pass
            elif knit_kind == 'revisions':
                yield from self._fetch_revision_texts(revs)
            else:
                raise AssertionError('Unknown knit kind %r' % knit_kind)

    def get_stream_for_missing_keys(self, missing_keys):
        keys = {}
        keys['texts'] = set()
        keys['revisions'] = set()
        keys['inventories'] = set()
        keys['chk_bytes'] = set()
        keys['signatures'] = set()
        for key in missing_keys:
            keys[key[0]].add(key[1:])
        if len(keys['revisions']):
            raise AssertionError('cannot copy revisions to fill in missing deltas {}'.format(keys['revisions']))
        for substream_kind, keys in keys.items():
            vf = getattr(self.from_repository, substream_kind)
            if vf is None and keys:
                raise AssertionError("cannot fill in keys for a versioned file we don't have: %s needs %s" % (substream_kind, keys))
            if not keys:
                continue
            if substream_kind == 'inventories':
                present = self.from_repository.inventories.get_parent_map(keys)
                revs = [key[0] for key in present]
                yield from self._get_inventory_stream(revs, missing=True)
                continue
            stream = versionedfile.filter_absent(vf.get_record_stream(keys, self.to_format._fetch_order, True))
            yield (substream_kind, stream)

    def inventory_fetch_order(self):
        if self._rich_root_upgrade():
            return 'topological'
        else:
            return self.to_format._fetch_order

    def _rich_root_upgrade(self):
        return not self.from_repository._format.rich_root_data and self.to_format.rich_root_data

    def _get_inventory_stream(self, revision_ids, missing=False):
        from_format = self.from_repository._format
        if from_format.supports_chks and self.to_format.supports_chks and (from_format.network_name() == self.to_format.network_name()):
            raise AssertionError('this case should be handled by GroupCHKStreamSource')
        elif 'forceinvdeltas' in debug.debug_flags:
            return self._get_convertable_inventory_stream(revision_ids, delta_versus_null=missing)
        elif from_format.network_name() == self.to_format.network_name():
            return self._get_simple_inventory_stream(revision_ids, missing=missing)
        elif not from_format.supports_chks and (not self.to_format.supports_chks) and (from_format._serializer == self.to_format._serializer):
            return self._get_simple_inventory_stream(revision_ids, missing=missing)
        else:
            return self._get_convertable_inventory_stream(revision_ids, delta_versus_null=missing)

    def _get_simple_inventory_stream(self, revision_ids, missing=False):
        from_weave = self.from_repository.inventories
        if missing:
            delta_closure = True
        else:
            delta_closure = not self.delta_on_metadata()
        yield ('inventories', from_weave.get_record_stream([(rev_id,) for rev_id in revision_ids], self.inventory_fetch_order(), delta_closure))

    def _get_convertable_inventory_stream(self, revision_ids, delta_versus_null=False):
        yield ('inventory-deltas', self._stream_invs_as_deltas(revision_ids, delta_versus_null=delta_versus_null))

    def _stream_invs_as_deltas(self, revision_ids, delta_versus_null=False):
        """Return a stream of inventory-deltas for the given rev ids.

        :param revision_ids: The list of inventories to transmit
        :param delta_versus_null: Don't try to find a minimal delta for this
            entry, instead compute the delta versus the NULL_REVISION. This
            effectively streams a complete inventory. Used for stuff like
            filling in missing parents, etc.
        """
        from_repo = self.from_repository
        revision_keys = [(rev_id,) for rev_id in revision_ids]
        parent_map = from_repo.inventories.get_parent_map(revision_keys)
        inventories = self.from_repository.iter_inventories(revision_ids, 'topological')
        format = from_repo._format
        invs_sent_so_far = {_mod_revision.NULL_REVISION}
        inventory_cache = lru_cache.LRUCache(50)
        null_inventory = from_repo.revision_tree(_mod_revision.NULL_REVISION).root_inventory
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=format.rich_root_data, tree_references=format.supports_tree_reference)
        for inv in inventories:
            key = (inv.revision_id,)
            parent_keys = parent_map.get(key, ())
            delta = None
            if not delta_versus_null and parent_keys:
                parent_ids = [parent_key[0] for parent_key in parent_keys]
                for parent_id in parent_ids:
                    if parent_id not in invs_sent_so_far:
                        continue
                    if parent_id == _mod_revision.NULL_REVISION:
                        parent_inv = null_inventory
                    else:
                        parent_inv = inventory_cache.get(parent_id, None)
                        if parent_inv is None:
                            parent_inv = from_repo.get_inventory(parent_id)
                    candidate_delta = inv._make_delta(parent_inv)
                    if delta is None or len(delta) > len(candidate_delta):
                        delta = candidate_delta
                        basis_id = parent_id
            if delta is None:
                basis_id = _mod_revision.NULL_REVISION
                delta = inv._make_delta(null_inventory)
            invs_sent_so_far.add(inv.revision_id)
            inventory_cache[inv.revision_id] = inv
            delta_serialized = serializer.delta_to_lines(basis_id, key[-1], delta)
            yield versionedfile.ChunkedContentFactory(key, parent_keys, None, delta_serialized, chunks_are_lines=True)