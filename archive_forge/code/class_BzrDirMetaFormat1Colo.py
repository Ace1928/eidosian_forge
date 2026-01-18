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
class BzrDirMetaFormat1Colo(BzrDirMetaFormat1):
    """BzrDirMeta1 format with support for colocated branches."""
    colocated_branches = True

    @classmethod
    def get_format_string(cls):
        """See BzrDirFormat.get_format_string()."""
        return b'Bazaar meta directory, format 1 (with colocated branches)\n'

    def get_format_description(self):
        """See BzrDirFormat.get_format_description()."""
        return 'Meta directory format 1 with support for colocated branches'

    def _open(self, transport):
        """See BzrDirFormat._open."""
        format = BzrDirMetaFormat1Colo()
        self._supply_sub_formats_to(format)
        return BzrDirMeta1(transport, format)