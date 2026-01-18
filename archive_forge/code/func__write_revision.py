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
def _write_revision(self, rev, rev_tree, base_rev, base_tree, explicit_base, force_binary):
    """Write out the information for a revision."""

    def w(key, value):
        self._write(key, value, indent=1)
    w('message', rev.message.split('\n'))
    w('committer', rev.committer)
    w('date', format_highres_date(rev.timestamp, rev.timezone))
    self.to_file.write(b'\n')
    self._write_delta(rev_tree, base_tree, rev.revision_id, force_binary)
    w('revision id', rev.revision_id)
    w('sha1', self._testament_sha1(rev.revision_id))
    w('inventory sha1', rev.inventory_sha1)
    if rev.parent_ids:
        w('parent ids', rev.parent_ids)
    if explicit_base:
        w('base id', base_rev)
    if rev.properties:
        self._write('properties', None, indent=1)
        for name, value in sorted(rev.properties.items()):
            self._write(name, value, indent=3, trailing_space_when_empty=True)
    self.to_file.write(b'\n')