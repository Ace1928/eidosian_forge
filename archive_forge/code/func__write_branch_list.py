import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def _write_branch_list(self, branches):
    """Write out the branch list.

        :param branches: List of utf-8 branch names to write
        """
    self.transport.put_bytes('branch-list', b''.join([name.encode('utf-8') + b'\n' for name in branches]))