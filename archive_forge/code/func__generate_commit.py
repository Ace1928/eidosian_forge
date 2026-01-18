import time
from io import BytesIO
from dulwich import __version__ as dulwich_version
from dulwich.objects import Blob
from .. import __version__ as brz_version
from .. import branch as _mod_branch
from .. import diff as _mod_diff
from .. import errors, osutils
from .. import revision as _mod_revision
from ..merge_directive import BaseMergeDirective
from .mapping import object_mode
from .object_store import get_object_store
@classmethod
def _generate_commit(cls, repository, revision_id, num, total, context=_mod_diff.DEFAULT_CONTEXT_AMOUNT):
    s = BytesIO()
    store = get_object_store(repository)
    with store.lock_read():
        commit = store[repository.lookup_bzr_revision_id(revision_id)[0]]
    from dulwich.patch import get_summary, write_commit_patch
    try:
        lhs_parent = repository.get_revision(revision_id).parent_ids[0]
    except IndexError:
        lhs_parent = _mod_revision.NULL_REVISION
    tree_1 = repository.revision_tree(lhs_parent)
    tree_2 = repository.revision_tree(revision_id)
    contents = BytesIO()
    differ = GitDiffTree.from_trees_options(tree_1, tree_2, contents, 'utf8', None, 'a/', 'b/', None, context_lines=context)
    differ.show_diff(None, None)
    write_commit_patch(s, commit, contents.getvalue(), (num, total), version_tail)
    summary = generate_patch_filename(num, get_summary(commit))
    return (summary, s.getvalue())