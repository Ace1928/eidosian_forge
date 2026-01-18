from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_delete(self, relpath):
    return self.to_transport.delete(urlutils.escape(relpath))