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
def _get_parent_map_with_sources(self, keys):
    """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A tuple. The first element is a mapping from keys to parents.
            Absent keys are absent from the mapping. The second element is a
            list with the locations each key was found in. The first element
            is the in-this-knit parents, the second the first fallback source,
            and so on.
        """
    result = {}
    sources = [self._index] + self._immediate_fallback_vfs
    source_results = []
    missing = set(keys)
    for source in sources:
        if not missing:
            break
        new_result = source.get_parent_map(missing)
        source_results.append(new_result)
        result.update(new_result)
        missing.difference_update(set(new_result))
    return (result, source_results)