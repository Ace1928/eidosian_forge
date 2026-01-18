import sys
import time
from io import StringIO
from . import branch as _mod_branch
from . import controldir, errors
from . import hooks as _mod_hooks
from . import osutils, urlutils
from .bzr import bzrdir
from .errors import (NoRepositoryPresent, NotBranchError, NotLocalUrl,
from .missing import find_unmerged
class LocationList:

    def __init__(self, base_path):
        self.locs = []
        self.base_path = base_path

    def add_url(self, label, url):
        """Add a URL to the list, converting it to a path if possible"""
        if url is None:
            return
        try:
            path = urlutils.local_path_from_url(url)
        except urlutils.InvalidURL:
            self.locs.append((label, url))
        else:
            self.add_path(label, path)

    def add_path(self, label, path):
        """Add a path, converting it to a relative path if possible"""
        try:
            path = osutils.relpath(self.base_path, path)
        except errors.PathNotChild:
            pass
        else:
            if path == '':
                path = '.'
        if path != '/':
            path = path.rstrip('/')
        self.locs.append((label, path))

    def get_lines(self):
        max_len = max((len(l) for l, u in self.locs))
        return ['  %*s: %s\n' % (max_len, l, u) for l, u in self.locs]