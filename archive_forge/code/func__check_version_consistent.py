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
def _check_version_consistent(self, other, other_idx, name):
    """Check if a version in consistent in this and other.

        To be consistent it must have:

         * the same text
         * the same direct parents (by name, not index, and disregarding
           order)

        If present & correct return True;
        if not present in self return False;
        if inconsistent raise error."""
    this_idx = self._name_map.get(name, -1)
    if this_idx != -1:
        if self._sha1s[this_idx] != other._sha1s[other_idx]:
            raise WeaveTextDiffers(name, self, other)
        self_parents = self._parents[this_idx]
        other_parents = other._parents[other_idx]
        n1 = {self._names[i] for i in self_parents}
        n2 = {other._names[i] for i in other_parents}
        if not self._compatible_parents(n1, n2):
            raise WeaveParentMismatch('inconsistent parents for version {%s}: %s vs %s' % (name, n1, n2))
        else:
            return True
    else:
        return False