from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class ColocatedDirectory(Directory):
    """Directory lookup for colocated branches.

    co:somename will resolve to the colocated branch with "somename" in
    the current directory.
    """

    def look_up(self, name, url, purpose=None):
        dir = _mod_controldir.ControlDir.open_containing('.')[0]
        return urlutils.join_segment_parameters(dir.user_url, {'branch': urlutils.escape(name)})