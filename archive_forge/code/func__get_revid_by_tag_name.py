from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def _get_revid_by_tag_name(self, tag_name):
    for branch in self.dir.list_branches():
        try:
            return branch.tags.lookup_tag(tag_name)
        except errors.NoSuchTag:
            pass
    return None