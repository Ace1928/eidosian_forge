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
def _read_footer(self):
    """Read the rest of the meta information.

        :param first_line:  The previous step iterates past what it
                            can handle. That extra line is given here.
        """
    for line in self._next():
        self._handle_next(line)
        if self._next_line is None:
            break
        if not self._next_line.startswith(b'#'):
            next(self._next())
            break