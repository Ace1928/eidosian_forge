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
def _copy_weave_content(self, otherweave):
    """adsorb the content from otherweave."""
    for attr in self.__slots__:
        if attr != '_weave_name':
            setattr(self, attr, copy(getattr(otherweave, attr)))