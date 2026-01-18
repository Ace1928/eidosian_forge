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
def _load_info_and_params(self):
    from .. import bzr_commit_handler
    repo_transport = self.repo.control_transport
    self.id_map_path = repo_transport.local_abspath('fastimport-id-map')
    info_path = self.params.get('info')
    if info_path is not None:
        self.info = configobj.ConfigObj(info_path)
    else:
        self.info = None
    self.supports_chk = self.repo._format.supports_chks
    self.commit_handler_factory = bzr_commit_handler.CommitHandler
    self.progress_every = _DEFAULT_AUTO_PROGRESS
    if self.verbose:
        self.progress_every = self.progress_every / 10
    self.checkpoint_every = int(self.params.get('checkpoint', _DEFAULT_AUTO_CHECKPOINT))
    self.checkpoint_count = 0
    self.autopack_every = int(self.params.get('autopack', _DEFAULT_AUTO_PACK))
    cache_size = int(self.params.get('inv-cache', -1))
    if cache_size == -1:
        if self.supports_chk:
            cache_size = _DEFAULT_CHK_INV_CACHE_SIZE
        else:
            cache_size = _DEFAULT_INV_CACHE_SIZE
    self.inventory_cache_size = cache_size
    try:
        self.max_commits = int(self.params['count'])
        if self.max_commits < 0:
            self.max_commits = None
    except KeyError:
        self.max_commits = None
    if self.info is not None:
        self.total_commits = int(self.info['Command counts']['commit'])
        if self.max_commits is not None and self.total_commits > self.max_commits:
            self.total_commits = self.max_commits
    else:
        self.total_commits = self.max_commits