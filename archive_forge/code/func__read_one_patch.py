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
def _read_one_patch(self):
    """Read in one patch, return the complete patch, along with
        the next line.

        :return: action, lines, do_continue
        """
    if self._next_line is None or self._next_line.startswith(b'#'):
        return (None, [], False)
    first = True
    lines = []
    for line in self._next():
        if first:
            if not line.startswith(b'==='):
                raise errors.MalformedPatches('The first line of all patches should be a bzr meta line "===": %r' % line)
            action = line[4:-1].decode('utf-8')
        elif line.startswith(b'... '):
            action += line[len(b'... '):-1].decode('utf-8')
        if self._next_line is not None and self._next_line.startswith(b'==='):
            return (action, lines, True)
        elif self._next_line is None or self._next_line.startswith(b'#'):
            return (action, lines, False)
        if first:
            first = False
        elif not line.startswith(b'... '):
            lines.append(line)
    return (action, lines, False)