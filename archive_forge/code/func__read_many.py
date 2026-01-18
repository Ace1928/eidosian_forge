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
def _read_many(self, indent):
    """If a line ends with no entry, that means that it should be
        followed with multiple lines of values.

        This detects the end of the list, because it will be a line that
        does not start properly indented.
        """
    values = []
    start = b'#' + b' ' * indent
    if self._next_line is None or not self._next_line.startswith(start):
        return values
    for line in self._next():
        values.append(line[len(start):-1].decode('utf-8'))
        if self._next_line is None or not self._next_line.startswith(start):
            break
    return values