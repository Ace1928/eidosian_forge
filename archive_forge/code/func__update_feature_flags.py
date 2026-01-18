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
def _update_feature_flags(self, updated_flags):
    """Update the feature flags in this format.

        :param updated_flags: Updated feature flags
        """
    for name, necessity in updated_flags.items():
        if necessity is None:
            try:
                del self.features[name]
            except KeyError:
                pass
        else:
            self.features[name] = necessity