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
class GCRepositoryPackCollection(RepositoryPackCollection):
    pack_factory = GCPack
    resumed_pack_factory = ResumedGCPack
    normal_packer_class = GCCHKPacker
    optimising_packer_class = GCCHKPacker

    def _check_new_inventories(self):
        """Detect missing inventories or chk root entries for the new revisions
        in this write group.

        :returns: list of strs, summarising any problems found.  If the list is
            empty no problems were found.
        """
        problems = []
        key_deps = self.repo.revisions._index._key_dependencies
        new_revisions_keys = key_deps.get_new_keys()
        no_fallback_inv_index = self.repo.inventories._index
        no_fallback_chk_bytes_index = self.repo.chk_bytes._index
        no_fallback_texts_index = self.repo.texts._index
        inv_parent_map = no_fallback_inv_index.get_parent_map(new_revisions_keys)
        corresponding_invs = set(inv_parent_map)
        missing_corresponding = set(new_revisions_keys)
        missing_corresponding.difference_update(corresponding_invs)
        if missing_corresponding:
            problems.append('inventories missing for revisions %s' % (sorted(missing_corresponding),))
            return problems
        all_inv_keys = set(corresponding_invs)
        for parent_inv_keys in inv_parent_map.values():
            all_inv_keys.update(parent_inv_keys)
        all_inv_keys.intersection_update(no_fallback_inv_index.get_parent_map(all_inv_keys))
        parent_invs_only_keys = all_inv_keys.symmetric_difference(corresponding_invs)
        all_missing = set()
        inv_ids = [key[-1] for key in all_inv_keys]
        parent_invs_only_ids = [key[-1] for key in parent_invs_only_keys]
        root_key_info = _build_interesting_key_sets(self.repo, inv_ids, parent_invs_only_ids)
        expected_chk_roots = root_key_info.all_keys()
        present_chk_roots = no_fallback_chk_bytes_index.get_parent_map(expected_chk_roots)
        missing_chk_roots = expected_chk_roots.difference(present_chk_roots)
        if missing_chk_roots:
            problems.append("missing referenced chk root keys: %s.Run 'brz reconcile --canonicalize-chks' on the affected repository." % (sorted(missing_chk_roots),))
            return problems
        chk_bytes_no_fallbacks = self.repo.chk_bytes.without_fallbacks()
        chk_bytes_no_fallbacks._search_key_func = self.repo.chk_bytes._search_key_func
        chk_diff = chk_map.iter_interesting_nodes(chk_bytes_no_fallbacks, root_key_info.interesting_root_keys, root_key_info.uninteresting_root_keys)
        text_keys = set()
        try:
            for record in _filter_text_keys(chk_diff, text_keys, chk_map._bytes_to_text_key):
                pass
        except errors.NoSuchRevision as e:
            problems.append('missing chk node(s) for id_to_entry maps')
        chk_diff = chk_map.iter_interesting_nodes(chk_bytes_no_fallbacks, root_key_info.interesting_pid_root_keys, root_key_info.uninteresting_pid_root_keys)
        try:
            for interesting_rec, interesting_map in chk_diff:
                pass
        except errors.NoSuchRevision as e:
            problems.append('missing chk node(s) for parent_id_basename_to_file_id maps')
        present_text_keys = no_fallback_texts_index.get_parent_map(text_keys)
        missing_text_keys = text_keys.difference(present_text_keys)
        if missing_text_keys:
            problems.append('missing text keys: %r' % (sorted(missing_text_keys),))
        return problems