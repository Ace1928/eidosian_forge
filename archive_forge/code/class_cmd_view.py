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
class cmd_view(Command):
    __doc__ = 'Manage filtered views.\n\n    Views provide a mask over the tree so that users can focus on\n    a subset of a tree when doing their work. After creating a view,\n    commands that support a list of files - status, diff, commit, etc -\n    effectively have that list of files implicitly given each time.\n    An explicit list of files can still be given but those files\n    must be within the current view.\n\n    In most cases, a view has a short life-span: it is created to make\n    a selected change and is deleted once that change is committed.\n    At other times, you may wish to create one or more named views\n    and switch between them.\n\n    To disable the current view without deleting it, you can switch to\n    the pseudo view called ``off``. This can be useful when you need\n    to see the whole tree for an operation or two (e.g. merge) but\n    want to switch back to your view after that.\n\n    :Examples:\n      To define the current view::\n\n        brz view file1 dir1 ...\n\n      To list the current view::\n\n        brz view\n\n      To delete the current view::\n\n        brz view --delete\n\n      To disable the current view without deleting it::\n\n        brz view --switch off\n\n      To define a named view and switch to it::\n\n        brz view --name view-name file1 dir1 ...\n\n      To list a named view::\n\n        brz view --name view-name\n\n      To delete a named view::\n\n        brz view --name view-name --delete\n\n      To switch to a named view::\n\n        brz view --switch view-name\n\n      To list all views defined::\n\n        brz view --all\n\n      To delete all views::\n\n        brz view --delete --all\n    '
    takes_args = ['file*']
    takes_options = [Option('all', help='Apply list or delete action to all views.'), Option('delete', help='Delete the view.'), Option('name', help='Name of the view to define, list or delete.', type=str), Option('switch', help='Name of the view to switch to.', type=str)]

    def run(self, file_list, all=False, delete=False, name=None, switch=None):
        tree, file_list = WorkingTree.open_containing_paths(file_list, apply_view=False)
        current_view, view_dict = tree.views.get_view_info()
        if name is None:
            name = current_view
        if delete:
            if file_list:
                raise errors.CommandError(gettext('Both --delete and a file list specified'))
            elif switch:
                raise errors.CommandError(gettext('Both --delete and --switch specified'))
            elif all:
                tree.views.set_view_info(None, {})
                self.outf.write(gettext('Deleted all views.\n'))
            elif name is None:
                raise errors.CommandError(gettext('No current view to delete'))
            else:
                tree.views.delete_view(name)
                self.outf.write(gettext("Deleted '%s' view.\n") % name)
        elif switch:
            if file_list:
                raise errors.CommandError(gettext('Both --switch and a file list specified'))
            elif all:
                raise errors.CommandError(gettext('Both --switch and --all specified'))
            elif switch == 'off':
                if current_view is None:
                    raise errors.CommandError(gettext('No current view to disable'))
                tree.views.set_view_info(None, view_dict)
                self.outf.write(gettext("Disabled '%s' view.\n") % current_view)
            else:
                tree.views.set_view_info(switch, view_dict)
                view_str = views.view_display_str(tree.views.lookup_view())
                self.outf.write(gettext("Using '{0}' view: {1}\n").format(switch, view_str))
        elif all:
            if view_dict:
                self.outf.write(gettext('Views defined:\n'))
                for view in sorted(view_dict):
                    if view == current_view:
                        active = '=>'
                    else:
                        active = '  '
                    view_str = views.view_display_str(view_dict[view])
                    self.outf.write('%s %-20s %s\n' % (active, view, view_str))
            else:
                self.outf.write(gettext('No views defined.\n'))
        elif file_list:
            if name is None:
                name = 'my'
            elif name == 'off':
                raise errors.CommandError(gettext("Cannot change the 'off' pseudo view"))
            tree.views.set_view(name, sorted(file_list))
            view_str = views.view_display_str(tree.views.lookup_view())
            self.outf.write(gettext("Using '{0}' view: {1}\n").format(name, view_str))
        elif name is None:
            self.outf.write(gettext('No current view.\n'))
        else:
            view_str = views.view_display_str(tree.views.lookup_view(name))
            self.outf.write(gettext("'{0}' view is: {1}\n").format(name, view_str))