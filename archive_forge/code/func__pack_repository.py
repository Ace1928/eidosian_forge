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
def _pack_repository(self, final=True):
    import gc
    if final:
        try:
            from .... import groupcompress
        except ImportError:
            pass
        else:
            groupcompress._FAST = False
    gc.collect()
    self.note('Packing repository ...')
    self.repo.pack()
    self.note('Removing obsolete packs ...')
    repo_transport = self.repo._pack_collection.transport
    obsolete_pack_transport = repo_transport.clone('obsolete_packs')
    for name in obsolete_pack_transport.list_dir('.'):
        obsolete_pack_transport.delete(name)
    if not final:
        gc.collect()