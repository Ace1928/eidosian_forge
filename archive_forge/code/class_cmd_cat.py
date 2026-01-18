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
class cmd_cat(Command):
    __doc__ = 'Write the contents of a file as of a given revision to standard output.\n\n    If no revision is nominated, the last revision is used.\n\n    Note: Take care to redirect standard output when using this command on a\n    binary file.\n    '
    _see_also = ['ls']
    takes_options = ['directory', Option('name-from-revision', help='The path name in the old tree.'), Option('filters', help='Apply content filters to display the convenience form.'), 'revision']
    takes_args = ['filename']
    encoding_type = 'exact'

    @display_command
    def run(self, filename, revision=None, name_from_revision=False, filters=False, directory=None):
        if revision is not None and len(revision) != 1:
            raise errors.CommandError(gettext('brz cat --revision takes exactly one revision specifier'))
        tree, branch, relpath = _open_directory_or_containing_tree_or_branch(filename, directory)
        self.enter_context(branch.lock_read())
        return self._run(tree, branch, relpath, filename, revision, name_from_revision, filters)

    def _run(self, tree, b, relpath, filename, revision, name_from_revision, filtered):
        import shutil
        if tree is None:
            tree = b.basis_tree()
        rev_tree = _get_one_revision_tree('cat', revision, branch=b)
        self.enter_context(rev_tree.lock_read())
        if name_from_revision:
            if not rev_tree.is_versioned(relpath):
                raise errors.CommandError(gettext('{0!r} is not present in revision {1}').format(filename, rev_tree.get_revision_id()))
            rev_tree_path = relpath
        else:
            try:
                rev_tree_path = _mod_tree.find_previous_path(tree, rev_tree, relpath)
            except transport.NoSuchFile:
                rev_tree_path = None
            if rev_tree_path is None:
                if not rev_tree.is_versioned(relpath):
                    raise errors.CommandError(gettext('{0!r} is not present in revision {1}').format(filename, rev_tree.get_revision_id()))
                else:
                    rev_tree_path = relpath
        if filtered:
            from .filter_tree import ContentFilterTree
            filter_tree = ContentFilterTree(rev_tree, rev_tree._content_filter_stack)
            fileobj = filter_tree.get_file(rev_tree_path)
        else:
            fileobj = rev_tree.get_file(rev_tree_path)
        shutil.copyfileobj(fileobj, self.outf)
        self.cleanup_now()