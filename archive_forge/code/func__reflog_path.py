import os
from .file import GitFile
from .index import commit_tree, iter_fresh_objects
from .reflog import drop_reflog_entry, read_reflog
@property
def _reflog_path(self):
    return os.path.join(self._repo.commondir(), 'logs', os.fsdecode(self._ref))