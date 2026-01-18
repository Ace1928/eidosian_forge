import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class BranchLoopError(errors.BzrError):
    """Encountered a branch cycle.

    A URL may point to a branch reference or it may point to a stacked branch.
    In either case, it's possible for there to be a cycle in these references,
    and this exception is raised when we detect such a cycle.
    """
    _fmt = 'Encountered a branch cycle'