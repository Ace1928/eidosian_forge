import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
@classmethod
def _init_maybe_bare(cls, path, controldir, bare, object_store=None, config=None, default_branch=None, symlinks: Optional[bool]=None):
    for d in BASE_DIRECTORIES:
        os.mkdir(os.path.join(controldir, *d))
    if object_store is None:
        object_store = DiskObjectStore.init(os.path.join(controldir, OBJECTDIR))
    ret = cls(path, bare=bare, object_store=object_store)
    if default_branch is None:
        if config is None:
            from .config import StackedConfig
            config = StackedConfig.default()
        try:
            default_branch = config.get('init', 'defaultBranch')
        except KeyError:
            default_branch = DEFAULT_BRANCH
    ret.refs.set_symbolic_ref(b'HEAD', LOCAL_BRANCH_PREFIX + default_branch)
    ret._init_files(bare=bare, symlinks=symlinks)
    return ret