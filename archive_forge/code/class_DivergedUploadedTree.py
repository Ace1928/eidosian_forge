from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
class DivergedUploadedTree(errors.CommandError):
    _fmt = 'Your branch (%(revid)s) and the uploaded tree (%(uploaded_revid)s) have diverged: '