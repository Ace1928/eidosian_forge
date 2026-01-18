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
class WeaveRevisionNotPresent(WeaveError):
    _fmt = 'Revision {%(revision_id)s} not present in %(weave)s'

    def __init__(self, revision_id, weave):
        WeaveError.__init__(self)
        self.revision_id = revision_id
        self.weave = weave