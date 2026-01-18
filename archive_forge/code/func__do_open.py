import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
def _do_open(self):
    basedir = os.path.dirname(self.path)
    if basedir:
        made_basedir = _ensure_tree(basedir)
        if made_basedir:
            self.logger.log(_utils.BLATHER, 'Created lock base path `%s`', basedir)
    if self.lockfile is None or self.lockfile.closed:
        self.lockfile = open(self.path, 'a')