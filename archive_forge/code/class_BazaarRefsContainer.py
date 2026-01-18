from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
class BazaarRefsContainer(RefsContainer):

    def __init__(self, dir, object_store):
        self.dir = dir
        self.object_store = object_store

    def get_packed_refs(self):
        return {}

    def set_symbolic_ref(self, name, other):
        if name == b'HEAD':
            pass
        else:
            raise NotImplementedError('Symbolic references not supported for anything other than HEAD')

    def _get_revid_by_tag_name(self, tag_name):
        for branch in self.dir.list_branches():
            try:
                return branch.tags.lookup_tag(tag_name)
            except errors.NoSuchTag:
                pass
        return None

    def _get_revid_by_branch_name(self, branch_name):
        try:
            branch = self.dir.open_branch(branch_name)
        except controldir.NoColocatedBranchSupport:
            if branch_name in ('HEAD', 'master'):
                branch = self.dir.open_branch()
            else:
                raise
        return branch.last_revision()

    def read_loose_ref(self, ref):
        try:
            branch_name = ref_to_branch_name(ref)
        except ValueError:
            tag_name = ref_to_tag_name(ref)
            revid = self._get_revid_by_tag_name(tag_name)
        else:
            revid = self._get_revid_by_branch_name(branch_name)
        if revid == _mod_revision.NULL_REVISION:
            return None
        with self.object_store.lock_read():
            return self.object_store._lookup_revision_sha1(revid)

    def get_peeled(self, ref):
        return self.read_loose_ref(ref)

    def allkeys(self):
        keys = set()
        for branch in self.dir.list_branches():
            repo = branch.repository
            if repo.has_revision(branch.last_revision()):
                ref = branch_name_to_ref(getattr(branch, 'name', ''))
                keys.add(ref)
            try:
                for tag_name, revid in branch.tags.get_tag_dict().items():
                    if repo.has_revision(revid):
                        keys.add(tag_name_to_ref(tag_name))
            except errors.TagsNotSupported:
                pass
        return keys

    def __delitem__(self, ref):
        try:
            branch_name = ref_to_branch_name(ref)
        except ValueError:
            return
        self.dir.destroy_branch(branch_name)

    def __setitem__(self, ref, sha):
        try:
            branch_name = ref_to_branch_name(ref)
        except ValueError:
            return
        try:
            target_branch = self.repo_dir.open_branch(branch_name)
        except errors.NotBranchError:
            target_branch = self.repo.create_branch(branch_name)
        rev_id = self.mapping.revision_id_foreign_to_bzr(sha)
        with target_branch.lock_write():
            target_branch.generate_revision_history(rev_id)