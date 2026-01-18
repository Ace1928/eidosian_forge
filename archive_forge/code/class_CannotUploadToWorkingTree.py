from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
class CannotUploadToWorkingTree(errors.CommandError):
    _fmt = 'Cannot upload to a bzr managed working tree: %(url)s".'