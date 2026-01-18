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
def _get_components_positions(self, keys, allow_missing=False):
    """Produce a map of position data for the components of keys.

        This data is intended to be used for retrieving the knit records.

        A dict of key to (record_details, index_memo, next, parents) is
        returned.

        * method is the way referenced data should be applied.
        * index_memo is the handle to pass to the data access to actually get
          the data
        * next is the build-parent of the version, or None for fulltexts.
        * parents is the version_ids of the parents of this version

        :param allow_missing: If True do not raise an error on a missing
            component, just ignore it.
        """
    component_data = {}
    pending_components = keys
    while pending_components:
        build_details = self._index.get_build_details(pending_components)
        current_components = set(pending_components)
        pending_components = set()
        for key, details in build_details.items():
            index_memo, compression_parent, parents, record_details = details
            if compression_parent is not None:
                pending_components.add(compression_parent)
            component_data[key] = self._build_details_to_components(details)
        missing = current_components.difference(build_details)
        if missing and (not allow_missing):
            raise errors.RevisionNotPresent(missing.pop(), self)
    return component_data