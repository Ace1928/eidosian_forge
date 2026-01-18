import os
from copy import copy
from io import BytesIO
import patiencediff
from ..lazy_import import lazy_import
from breezy import tsort
from .. import errors, osutils
from .. import transport as _mod_transport
from ..errors import RevisionAlreadyPresent, RevisionNotPresent
from ..osutils import dirname, sha, sha_strings, split_lines
from ..revision import NULL_REVISION
from ..trace import mutter
from .versionedfile import (AbsentContentFactory, ContentFactory,
from .weavefile import _read_weave_v5, write_weave_v5
def _imported_parents(self, other, other_idx):
    """Return list of parents in self corresponding to indexes in other."""
    new_parents = []
    for parent_idx in other._parents[other_idx]:
        parent_name = other._names[parent_idx]
        if parent_name not in self._name_map:
            raise WeaveError('missing parent {%s} of {%s} in %r' % (parent_name, other._name_map[other_idx], self))
        new_parents.append(self._name_map[parent_name])
    return new_parents