import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _create_network_bytes(self):
    """Create a fully serialised network version for transmission."""
    key_bytes = b'\x00'.join(self.key)
    if self.parents is None:
        parent_bytes = b'None:'
    else:
        parent_bytes = b'\t'.join((b'\x00'.join(key) for key in self.parents))
    if self._build_details[1]:
        noeol = b'N'
    else:
        noeol = b' '
    network_bytes = b'%s\n%s\n%s\n%s%s' % (self.storage_kind.encode('ascii'), key_bytes, parent_bytes, noeol, self._raw_record)
    self._network_bytes = network_bytes