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
class cmd_ls(Command):
    __doc__ = 'List files in a tree.\n    '
    _see_also = ['status', 'cat']
    takes_args = ['path?']
    takes_options = ['verbose', 'revision', Option('recursive', short_name='R', help='Recurse into subdirectories.'), Option('from-root', help='Print paths relative to the root of the branch.'), Option('unknown', short_name='u', help='Print unknown files.'), Option('versioned', help='Print versioned files.', short_name='V'), Option('ignored', short_name='i', help='Print ignored files.'), Option('kind', short_name='k', help='List entries of a particular kind: file, directory, symlink, tree-reference.', type=str), 'null', 'show-ids', 'directory']

    @display_command
    def run(self, revision=None, verbose=False, recursive=False, from_root=False, unknown=False, versioned=False, ignored=False, null=False, kind=None, show_ids=False, path=None, directory=None):
        if kind and kind not in ('file', 'directory', 'symlink', 'tree-reference'):
            raise errors.CommandError(gettext('invalid kind specified'))
        if verbose and null:
            raise errors.CommandError(gettext('Cannot set both --verbose and --null'))
        all = not (unknown or versioned or ignored)
        selection = {'I': ignored, '?': unknown, 'V': versioned}
        if path is None:
            fs_path = '.'
        else:
            if from_root:
                raise errors.CommandError(gettext('cannot specify both --from-root and PATH'))
            fs_path = path
        tree, branch, relpath = _open_directory_or_containing_tree_or_branch(fs_path, directory)
        prefix = None
        if from_root:
            if relpath:
                prefix = relpath + '/'
        elif fs_path != '.' and (not fs_path.endswith('/')):
            prefix = fs_path + '/'
        if revision is not None or tree is None:
            tree = _get_one_revision_tree('ls', revision, branch=branch)
        apply_view = False
        if isinstance(tree, WorkingTree) and tree.supports_views():
            view_files = tree.views.lookup_view()
            if view_files:
                apply_view = True
                view_str = views.view_display_str(view_files)
                note(gettext('Ignoring files outside view. View is %s') % view_str)
        self.enter_context(tree.lock_read())
        for fp, fc, fkind, entry in tree.list_files(include_root=False, from_dir=relpath, recursive=recursive):
            if not all and (not selection[fc]):
                continue
            if kind is not None and fkind != kind:
                continue
            if apply_view:
                try:
                    if relpath:
                        fullpath = osutils.pathjoin(relpath, fp)
                    else:
                        fullpath = fp
                    views.check_path_in_view(tree, fullpath)
                except views.FileOutsideView:
                    continue
            if prefix:
                fp = osutils.pathjoin(prefix, fp)
            kindch = entry.kind_character()
            outstring = fp + kindch
            ui.ui_factory.clear_term()
            if verbose:
                outstring = '%-8s %s' % (fc, outstring)
                if show_ids and getattr(entry, 'file_id', None) is not None:
                    outstring = '%-50s %s' % (outstring, entry.file_id.decode('utf-8'))
                self.outf.write(outstring + '\n')
            elif null:
                self.outf.write(fp + '\x00')
                if show_ids:
                    if getattr(entry, 'file_id', None) is not None:
                        self.outf.write(entry.file_id.decode('utf-8'))
                    self.outf.write('\x00')
                self.outf.flush()
            elif show_ids:
                if getattr(entry, 'file_id', None) is not None:
                    my_id = entry.file_id.decode('utf-8')
                else:
                    my_id = ''
                self.outf.write('%-50s %s\n' % (outstring, my_id))
            else:
                self.outf.write(outstring + '\n')