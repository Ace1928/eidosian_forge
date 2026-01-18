from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_delete_tree(self, relpath):
    return self.to_transport.delete_tree(urlutils.escape(relpath))