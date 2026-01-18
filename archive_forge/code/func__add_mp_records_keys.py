import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def _add_mp_records_keys(self, repo_kind, vf, keys):
    """Add multi-parent diff records to a bundle"""
    ordered_keys = list(multiparent.topo_iter_keys(vf, keys))
    mpdiffs = vf.make_mpdiffs(ordered_keys)
    sha1s = vf.get_sha1s(ordered_keys)
    parent_map = vf.get_parent_map(ordered_keys)
    for mpdiff, item_key in zip(mpdiffs, ordered_keys):
        sha1 = sha1s[item_key]
        parents = [key[-1] for key in parent_map[item_key]]
        text = b''.join(mpdiff.to_patch())
        if len(item_key) == 2:
            file_id = item_key[0]
        else:
            file_id = None
        self.bundle.add_multiparent_record(text, sha1, parents, repo_kind, item_key[-1], file_id)