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
def _write_action(self, name, parameters, properties=None):
    if properties is None:
        properties = []
    p_texts = ['%s:%s' % v for v in properties]
    self.to_file.write(b'=== ')
    self.to_file.write(' '.join([name] + parameters).encode('utf-8'))
    self.to_file.write(' // '.join(p_texts).encode('utf-8'))
    self.to_file.write(b'\n')