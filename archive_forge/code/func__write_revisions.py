from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
def _write_revisions(self, pb):
    """Write the information for all of the revisions."""
    last_rev_id = None
    last_rev_tree = None
    i_max = len(self.revision_ids)
    for i, rev_id in enumerate(self.revision_ids):
        pb.update('Generating revision data', i, i_max)
        rev = self.source.get_revision(rev_id)
        if rev_id == last_rev_id:
            rev_tree = last_rev_tree
        else:
            rev_tree = self.source.revision_tree(rev_id)
        if rev_id in self.forced_bases:
            explicit_base = True
            base_id = self.forced_bases[rev_id]
            if base_id is None:
                base_id = NULL_REVISION
        else:
            explicit_base = False
            if rev.parent_ids:
                base_id = rev.parent_ids[-1]
            else:
                base_id = NULL_REVISION
        if base_id == last_rev_id:
            base_tree = last_rev_tree
        else:
            base_tree = self.source.revision_tree(base_id)
        force_binary = i != 0
        self._write_revision(rev, rev_tree, base_id, base_tree, explicit_base, force_binary)
        last_rev_id = base_id
        last_rev_tree = base_tree