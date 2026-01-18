from .. import debug, errors
from .. import revision as _mod_revision
from ..branch import Branch
from ..trace import mutter_callsite
from .branch import BranchFormatMetadir, BzrBranch
class BzrBranch5(FullHistoryBzrBranch):
    """A format 5 branch. This supports new features over plain branches.

    It has support for a master_branch which is the data for bound branches.
    """