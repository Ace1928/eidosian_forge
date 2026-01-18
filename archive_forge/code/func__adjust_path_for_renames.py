import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def _adjust_path_for_renames(self, path, renamed, revision_id):
    for old, new in renamed:
        if path == old:
            self.note('Changing path %s given rename to %s in revision %s' % (path, new, revision_id))
            path = new
        elif path.startswith(old + '/'):
            self.note('Adjusting path %s given rename of %s to %s in revision %s' % (path, old, new, revision_id))
            path = path.replace(old + '/', new + '/')
    return path