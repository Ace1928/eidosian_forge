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
class cmd_branch(Command):
    __doc__ = 'Create a new branch that is a copy of an existing branch.\n\n    If the TO_LOCATION is omitted, the last component of the FROM_LOCATION will\n    be used.  In other words, "branch ../foo/bar" will attempt to create ./bar.\n    If the FROM_LOCATION has no / or path separator embedded, the TO_LOCATION\n    is derived from the FROM_LOCATION by stripping a leading scheme or drive\n    identifier, if any. For example, "branch lp:foo-bar" will attempt to\n    create ./foo-bar.\n\n    To retrieve the branch as of a particular revision, supply the --revision\n    parameter, as in "branch foo/bar -r 5".\n    '
    aliase = ['sprout']
    _see_also = ['checkout']
    takes_args = ['from_location', 'to_location?']
    takes_options = ['revision', Option('hardlink', help='Hard-link working tree files where possible.'), Option('files-from', type=str, help='Get file contents from this tree.'), Option('no-tree', help='Create a branch without a working-tree.'), Option('switch', help='Switch the checkout in the current directory to the new branch.'), Option('stacked', help='Create a stacked branch referring to the source branch. The new branch will depend on the availability of the source branch for all operations.'), Option('standalone', help='Do not use a shared repository, even if available.'), Option('use-existing-dir', help='By default branch will fail if the target directory exists, but does not already have a control directory.  This flag will allow branch to proceed.'), Option('bind', help='Bind new branch to from location.'), Option('no-recurse-nested', help='Do not recursively check out nested trees.'), Option('colocated-branch', short_name='b', type=str, help='Name of colocated branch to sprout.')]

    def run(self, from_location, to_location=None, revision=None, hardlink=False, stacked=False, standalone=False, no_tree=False, use_existing_dir=False, switch=False, bind=False, files_from=None, no_recurse_nested=False, colocated_branch=None):
        from breezy import switch as _mod_switch
        accelerator_tree, br_from = controldir.ControlDir.open_tree_or_branch(from_location, name=colocated_branch)
        if no_recurse_nested:
            recurse = 'none'
        else:
            recurse = 'down'
        if not (hardlink or files_from):
            accelerator_tree = None
        if files_from is not None and files_from != from_location:
            accelerator_tree = WorkingTree.open(files_from)
        revision = _get_one_revision('branch', revision)
        self.enter_context(br_from.lock_read())
        if revision is not None:
            revision_id = revision.as_revision_id(br_from)
        else:
            revision_id = br_from.last_revision()
        if to_location is None:
            to_location = urlutils.derive_to_location(from_location)
        to_transport = transport.get_transport(to_location, purpose='write')
        try:
            to_transport.mkdir('.')
        except transport.FileExists:
            try:
                to_dir = controldir.ControlDir.open_from_transport(to_transport)
            except errors.NotBranchError as exc:
                if not use_existing_dir:
                    raise errors.CommandError(gettext('Target directory "%s" already exists.') % to_location) from exc
                else:
                    to_dir = None
            else:
                try:
                    to_dir.open_branch()
                except errors.NotBranchError:
                    pass
                else:
                    raise errors.AlreadyBranchError(to_location)
        except transport.NoSuchFile as exc:
            raise errors.CommandError(gettext('Parent of "%s" does not exist.') % to_location) from exc
        else:
            to_dir = None
        if to_dir is None:
            try:
                to_dir = br_from.controldir.sprout(to_transport.base, revision_id, possible_transports=[to_transport], accelerator_tree=accelerator_tree, hardlink=hardlink, stacked=stacked, force_new_repo=standalone, create_tree_if_local=not no_tree, source_branch=br_from, recurse=recurse)
                branch = to_dir.open_branch(possible_transports=[br_from.controldir.root_transport, to_transport])
            except errors.NoSuchRevision as exc:
                to_transport.delete_tree('.')
                msg = gettext('The branch {0} has no revision {1}.').format(from_location, revision)
                raise errors.CommandError(msg) from exc
        else:
            try:
                to_repo = to_dir.open_repository()
            except errors.NoRepositoryPresent:
                to_repo = to_dir.create_repository()
            to_repo.fetch(br_from.repository, revision_id=revision_id)
            branch = br_from.sprout(to_dir, revision_id=revision_id)
        br_from.tags.merge_to(branch.tags)
        try:
            note(gettext('Created new stacked branch referring to %s.') % branch.get_stacked_on_url())
        except (errors.NotStacked, _mod_branch.UnstackableBranchFormat, errors.UnstackableRepositoryFormat) as e:
            revno = branch.revno()
            if revno is not None:
                note(ngettext('Branched %d revision.', 'Branched %d revisions.', branch.revno()) % revno)
            else:
                note(gettext('Created new branch.'))
        if bind:
            parent_branch = Branch.open(from_location)
            branch.bind(parent_branch)
            note(gettext('New branch bound to %s') % from_location)
        if switch:
            wt, _ = WorkingTree.open_containing('.')
            _mod_switch.switch(wt.controldir, branch)
            note(gettext('Switched to branch: %s'), urlutils.unescape_for_display(branch.base, 'utf-8'))