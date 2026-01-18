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
def _get_parent_annotations_and_matches(self, key, text, parent_key):
    """Get the list of annotations for the parent, and the matching lines.

        :param text: The opaque value given by _get_needed_texts
        :param parent_key: The key for the parent text
        :return: (parent_annotations, matching_blocks)
            parent_annotations is a list as long as the number of lines in
                parent
            matching_blocks is a list of (parent_idx, text_idx, len) tuples
                indicating which lines match between the two texts
        """
    block_key = (key, parent_key)
    if block_key in self._matching_blocks:
        blocks = self._matching_blocks.pop(block_key)
        parent_annotations = self._annotations_cache[parent_key]
        return (parent_annotations, blocks)
    return annotate.Annotator._get_parent_annotations_and_matches(self, key, text, parent_key)