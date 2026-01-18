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
def _sort_keys_by_io(self, keys, positions):
    """Figure out an optimal order to read the records for the given keys.

        Sort keys, grouped by index and sorted by position.

        :param keys: A list of keys whose records we want to read. This will be
            sorted 'in-place'.
        :param positions: A dict, such as the one returned by
            _get_components_positions()
        :return: None
        """

    def get_index_memo(key):
        return positions[key][1]
    return keys.sort(key=get_index_memo)