import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def _check_permissions(self):
    """Check permission of auth file are user read/write able only."""
    import stat
    try:
        st = os.stat(self._filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            trace.mutter('Unable to stat %r: %r', self._filename, e)
        return
    mode = stat.S_IMODE(st.st_mode)
    if (stat.S_IXOTH | stat.S_IWOTH | stat.S_IROTH | stat.S_IXGRP | stat.S_IWGRP | stat.S_IRGRP) & mode:
        if self._filename not in _authentication_config_permission_errors and (not GlobalConfig().suppress_warning('insecure_permissions')):
            trace.warning("The file '%s' has insecure file permissions. Saved passwords may be accessible by other users.", self._filename)
            _authentication_config_permission_errors.add(self._filename)