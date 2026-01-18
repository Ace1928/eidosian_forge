from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_rename(self, old_path, new_path):
    return self.to_transport.rename(urlutils.escape(old_path), urlutils.escape(new_path))