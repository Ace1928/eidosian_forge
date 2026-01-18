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
def _handle_next(self, line):
    if line is None:
        return
    key, value = self._read_next_entry(line, indent=1)
    mutter('_handle_next {!r} => {!r}'.format(key, value))
    if key is None:
        return
    revision_info = self.info.revisions[-1]
    if key in revision_info.__dict__:
        if getattr(revision_info, key) is None:
            if key in ('file_id', 'revision_id', 'base_id'):
                value = value.encode('utf8')
            elif key in 'parent_ids':
                value = [v.encode('utf8') for v in value]
            elif key in ('testament_sha1', 'inventory_sha1', 'sha1'):
                value = value.encode('ascii')
            setattr(revision_info, key, value)
        else:
            raise errors.MalformedHeader('Duplicated Key: %s' % key)
    else:
        raise errors.MalformedHeader('Unknown Key: "%s"' % key)