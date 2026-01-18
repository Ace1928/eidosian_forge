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
def _read_revision_header(self):
    found_something = False
    self.info.revisions.append(RevisionInfo(None))
    for line in self._next():
        if line is None or line == b'\n':
            break
        if not line.startswith(b'#'):
            continue
        found_something = True
        self._handle_next(line)
    if not found_something:
        self.info.revisions.pop()
    return found_something