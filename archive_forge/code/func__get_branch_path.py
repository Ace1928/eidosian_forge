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
def _get_branch_path(self, name):
    """Obtain the branch path to use.

        This uses the API specified branch name first, and then falls back to
        the branch name specified in the URL. If neither of those is specified,
        it uses the default branch.

        :param name: Optional branch name to use
        :return: Relative path to branch
        """
    if name == '':
        return 'branch'
    return urlutils.join('branches', urlutils.escape(name))