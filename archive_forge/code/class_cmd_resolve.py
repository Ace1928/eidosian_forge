import errno
import os
import re
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import commands, errors, option, osutils, registry, trace
class cmd_resolve(commands.Command):
    __doc__ = 'Mark a conflict as resolved.\n\n    Merge will do its best to combine the changes in two branches, but there\n    are some kinds of problems only a human can fix.  When it encounters those,\n    it will mark a conflict.  A conflict means that you need to fix something,\n    before you can commit.\n\n    Once you have fixed a problem, use "brz resolve" to automatically mark\n    text conflicts as fixed, "brz resolve FILE" to mark a specific conflict as\n    resolved, or "brz resolve --all" to mark all conflicts as resolved.\n    '
    aliases = ['resolved']
    takes_args = ['file*']
    takes_options = ['directory', option.Option('all', help='Resolve all conflicts in this tree.'), ResolveActionOption()]
    _see_also = ['conflicts']

    def run(self, file_list=None, all=False, action=None, directory=None):
        if all:
            if file_list:
                raise errors.CommandError(gettext('If --all is specified, no FILE may be provided'))
            if directory is None:
                directory = '.'
            tree = workingtree.WorkingTree.open_containing(directory)[0]
            if action is None:
                action = 'done'
        else:
            tree, file_list = workingtree.WorkingTree.open_containing_paths(file_list, directory)
            if action is None:
                if file_list is None:
                    action = 'auto'
                else:
                    action = 'done'
        before, after = resolve(tree, file_list, action=action)
        if action == 'auto' and file_list is None:
            if after > 0:
                trace.note(ngettext('%d conflict auto-resolved.', '%d conflicts auto-resolved.', before - after), before - after)
                trace.note(gettext('Remaining conflicts:'))
                for conflict in tree.conflicts():
                    trace.note(str(conflict))
                return 1
            else:
                trace.note(gettext('All conflicts resolved.'))
                return 0
        else:
            trace.note(ngettext('{0} conflict resolved, {1} remaining', '{0} conflicts resolved, {1} remaining', before - after).format(before - after, after))