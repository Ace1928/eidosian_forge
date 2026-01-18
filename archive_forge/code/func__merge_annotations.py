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
def _merge_annotations(self, content, parents, parent_texts={}, delta=None, annotated=None, left_matching_blocks=None):
    """Merge annotations for content and generate deltas.

        This is done by comparing the annotations based on changes to the text
        and generating a delta on the resulting full texts. If annotations are
        not being created then a simple delta is created.
        """
    if left_matching_blocks is not None:
        delta_seq = diff._PrematchedMatcher(left_matching_blocks)
    else:
        delta_seq = None
    if annotated:
        for parent_key in parents:
            merge_content = self._get_content(parent_key, parent_texts)
            if parent_key == parents[0] and delta_seq is not None:
                seq = delta_seq
            else:
                seq = patiencediff.PatienceSequenceMatcher(None, merge_content.text(), content.text())
            for i, j, n in seq.get_matching_blocks():
                if n == 0:
                    continue
                content._lines[j:j + n] = merge_content._lines[i:i + n]
        if content._lines and (not content._lines[-1][1].endswith(b'\n')):
            line = content._lines[-1][1] + b'\n'
            content._lines[-1] = (content._lines[-1][0], line)
    if delta:
        if delta_seq is None:
            reference_content = self._get_content(parents[0], parent_texts)
            new_texts = content.text()
            old_texts = reference_content.text()
            delta_seq = patiencediff.PatienceSequenceMatcher(None, old_texts, new_texts)
        return self._make_line_delta(delta_seq, content)