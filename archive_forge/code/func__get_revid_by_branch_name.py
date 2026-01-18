from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def _get_revid_by_branch_name(self, branch_name):
    try:
        branch = self.dir.open_branch(branch_name)
    except controldir.NoColocatedBranchSupport:
        if branch_name in ('HEAD', 'master'):
            branch = self.dir.open_branch()
        else:
            raise
    return branch.last_revision()