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
def _read_next_entry(self, line, indent=1):
    """Read in a key-value pair
        """
    if not line.startswith(b'#'):
        raise errors.MalformedHeader('Bzr header did not start with #')
    line = line[1:-1].decode('utf-8')
    if line[:indent] == ' ' * indent:
        line = line[indent:]
    if not line:
        return (None, None)
    loc = line.find(': ')
    if loc != -1:
        key = line[:loc]
        value = line[loc + 2:]
        if not value:
            value = self._read_many(indent=indent + 2)
    elif line[-1:] == ':':
        key = line[:-1]
        value = self._read_many(indent=indent + 2)
    else:
        raise errors.MalformedHeader('While looking for key: value pairs, did not find the colon %r' % line)
    key = key.replace(' ', '_')
    return (key, value)