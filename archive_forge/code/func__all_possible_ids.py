import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
def _all_possible_ids(self):
    """Return all the possible revisions that we could find."""
    if 'evil' in debug.debug_flags:
        trace.mutter_callsite(3, '_all_possible_ids scales with size of history.')
    with self.lock_read():
        return [key[-1] for key in self.inventories.keys()]