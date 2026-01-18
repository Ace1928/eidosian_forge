import time
import configobj
from fastimport import commands
from fastimport import errors as plugin_errors
from fastimport import processor
from fastimport.helpers import invert_dictset
from .... import debug, delta, errors, osutils, progress
from .... import revision as _mod_revision
from ....bzr.knitpack_repo import KnitPackRepository
from ....trace import mutter, note, warning
from .. import (branch_updater, cache_manager, helpers, idmapfile, marks_file,
def _init_id_map(self):
    """Load the id-map and check it matches the repository.

        :return: the number of entries in the map
        """
    self.cache_mgr.marks, known = idmapfile.load_id_map(self.id_map_path)
    if self.cache_mgr.add_mark(b'0', _mod_revision.NULL_REVISION):
        known += 1
    existing_count = len(self.repo.all_revision_ids())
    if existing_count < known:
        raise plugin_errors.BadRepositorySize(known, existing_count)
    return known