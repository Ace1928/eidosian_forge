import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_update(Command):
    __doc__ = "Update a working tree to a new revision.\n\n    This will perform a merge of the destination revision (the tip of the\n    branch, or the specified revision) into the working tree, and then make\n    that revision the basis revision for the working tree.\n\n    You can use this to visit an older revision, or to update a working tree\n    that is out of date from its branch.\n\n    If there are any uncommitted changes in the tree, they will be carried\n    across and remain as uncommitted changes after the update.  To discard\n    these changes, use 'brz revert'.  The uncommitted changes may conflict\n    with the changes brought in by the change in basis revision.\n\n    If the tree's branch is bound to a master branch, brz will also update\n    the branch from the master.\n\n    You cannot update just a single file or directory, because each Breezy\n    working tree has just a single basis revision.  If you want to restore a\n    file that has been removed locally, use 'brz revert' instead of 'brz\n    update'.  If you want to restore a file to its state in a previous\n    revision, use 'brz revert' with a '-r' option, or use 'brz cat' to write\n    out the old content of that file to a new location.\n\n    The 'dir' argument, if given, must be the location of the root of a\n    working tree to update.  By default, the working tree that contains the\n    current working directory is used.\n    "
    _see_also = ['pull', 'working-trees', 'status-flags']
    takes_args = ['dir?']
    takes_options = ['revision', Option('show-base', help='Show base revision text in conflicts.')]
    aliases = ['up']

    def run(self, dir=None, revision=None, show_base=None):
        if revision is not None and len(revision) != 1:
            raise errors.CommandError(gettext('brz update --revision takes exactly one revision'))
        if dir is None:
            tree = WorkingTree.open_containing('.')[0]
        else:
            tree, relpath = WorkingTree.open_containing(dir)
            if relpath:
                raise errors.CommandError(gettext('brz update can only update a whole tree, not a file or subdirectory'))
        branch = tree.branch
        possible_transports = []
        master = branch.get_master_branch(possible_transports=possible_transports)
        if master is not None:
            branch_location = master.base
            self.enter_context(tree.lock_write())
        else:
            branch_location = tree.branch.base
            self.enter_context(tree.lock_tree_write())
        branch_location = urlutils.unescape_for_display(branch_location.rstrip('/'), self.outf.encoding)
        existing_pending_merges = tree.get_parent_ids()[1:]
        if master is None:
            old_tip = None
        else:
            old_tip = branch.update(possible_transports)
        if revision is not None:
            revision_id = revision[0].as_revision_id(branch)
        else:
            revision_id = branch.last_revision()
        if revision_id == tree.last_revision():
            revno = branch.revision_id_to_dotted_revno(revision_id)
            note(gettext('Tree is up to date at revision {0} of branch {1}').format('.'.join(map(str, revno)), branch_location))
            return 0
        view_info = _get_view_info_for_change_reporter(tree)
        change_reporter = delta._ChangeReporter(unversioned_filter=tree.is_ignored, view_info=view_info)
        try:
            conflicts = tree.update(change_reporter, possible_transports=possible_transports, revision=revision_id, old_tip=old_tip, show_base=show_base)
        except errors.NoSuchRevision as exc:
            raise errors.CommandError(gettext('branch has no revision %s\nbrz update --revision only works for a revision in the branch history') % exc.revision) from exc
        revno = tree.branch.revision_id_to_dotted_revno(tree.last_revision())
        note(gettext('Updated to revision {0} of branch {1}').format('.'.join(map(str, revno)), branch_location))
        parent_ids = tree.get_parent_ids()
        if parent_ids[1:] and parent_ids[1:] != existing_pending_merges:
            note(gettext("Your local commits will now show as pending merges with 'brz status', and can be committed with 'brz commit'."))
        if conflicts != 0:
            return 1
        else:
            return 0