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
class cmd_mv(Command):
    __doc__ = 'Move or rename a file.\n\n    :Usage:\n        brz mv OLDNAME NEWNAME\n\n        brz mv SOURCE... DESTINATION\n\n    If the last argument is a versioned directory, all the other names\n    are moved into it.  Otherwise, there must be exactly two arguments\n    and the file is changed to a new name.\n\n    If OLDNAME does not exist on the filesystem but is versioned and\n    NEWNAME does exist on the filesystem but is not versioned, mv\n    assumes that the file has been manually moved and only updates\n    its internal inventory to reflect that change.\n    The same is valid when moving many SOURCE files to a DESTINATION.\n\n    Files cannot be moved between branches.\n    '
    takes_args = ['names*']
    takes_options = [Option('after', help='Move only the brz identifier of the file, because the file has already been moved.'), Option('auto', help='Automatically guess renames.'), Option('dry-run', help='Avoid making changes when guessing renames.')]
    aliases = ['move', 'rename']
    encoding_type = 'replace'

    def run(self, names_list, after=False, auto=False, dry_run=False):
        if auto:
            return self.run_auto(names_list, after, dry_run)
        elif dry_run:
            raise errors.CommandError(gettext('--dry-run requires --auto.'))
        if names_list is None:
            names_list = []
        if len(names_list) < 2:
            raise errors.CommandError(gettext('missing file argument'))
        tree, rel_names = WorkingTree.open_containing_paths(names_list, canonicalize=False)
        for file_name in rel_names[0:-1]:
            if file_name == '':
                raise errors.CommandError(gettext('can not move root of branch'))
        self.enter_context(tree.lock_tree_write())
        self._run(tree, names_list, rel_names, after)

    def run_auto(self, names_list, after, dry_run):
        if names_list is not None and len(names_list) > 1:
            raise errors.CommandError(gettext('Only one path may be specified to --auto.'))
        if after:
            raise errors.CommandError(gettext('--after cannot be specified with --auto.'))
        work_tree, file_list = WorkingTree.open_containing_paths(names_list, default_directory='.')
        self.enter_context(work_tree.lock_tree_write())
        rename_map.RenameMap.guess_renames(work_tree.basis_tree(), work_tree, dry_run)

    def _run(self, tree, names_list, rel_names, after):
        into_existing = osutils.isdir(names_list[-1])
        if into_existing and len(names_list) == 2:
            if not tree.case_sensitive and rel_names[0].lower() == rel_names[1].lower():
                into_existing = False
            else:
                from_path = tree.get_canonical_path(rel_names[0])
                if not osutils.lexists(names_list[0]) and tree.is_versioned(from_path) and (tree.stored_kind(from_path) == 'directory'):
                    into_existing = False
        if into_existing:
            rel_names = list(tree.get_canonical_paths(rel_names))
            for src, dest in tree.move(rel_names[:-1], rel_names[-1], after=after):
                if not is_quiet():
                    self.outf.write('{} => {}\n'.format(src, dest))
        else:
            if len(names_list) != 2:
                raise errors.CommandError(gettext('to mv multiple files the destination must be a versioned directory'))
            src = tree.get_canonical_path(rel_names[0])
            canon_dest = tree.get_canonical_path(rel_names[1])
            dest_parent = osutils.dirname(canon_dest)
            spec_tail = osutils.basename(rel_names[1])
            dest_id = tree.path2id(canon_dest)
            if dest_id is None or tree.path2id(src) == dest_id:
                if after:
                    if dest_parent:
                        dest_parent_fq = osutils.pathjoin(tree.basedir, dest_parent)
                    else:
                        dest_parent_fq = tree.basedir
                    dest_tail = osutils.canonical_relpath(dest_parent_fq, osutils.pathjoin(dest_parent_fq, spec_tail))
                else:
                    dest_tail = spec_tail
            else:
                dest_tail = os.path.basename(canon_dest)
            dest = osutils.pathjoin(dest_parent, dest_tail)
            mutter('attempting to move %s => %s', src, dest)
            tree.rename_one(src, dest, after=after)
            if not is_quiet():
                self.outf.write('{} => {}\n'.format(src, dest))