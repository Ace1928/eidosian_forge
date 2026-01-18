import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class BranchReferenceForbidden(errors.BzrError):
    _fmt = 'Trying to mirror a branch reference and the branch type does not allow references.'