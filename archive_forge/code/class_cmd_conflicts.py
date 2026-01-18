import errno
import os
import re
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import commands, errors, option, osutils, registry, trace
class cmd_conflicts(commands.Command):
    __doc__ = 'List files with conflicts.\n\n    Merge will do its best to combine the changes in two branches, but there\n    are some kinds of problems only a human can fix.  When it encounters those,\n    it will mark a conflict.  A conflict means that you need to fix something,\n    before you can commit.\n\n    Conflicts normally are listed as short, human-readable messages.  If --text\n    is supplied, the pathnames of files with text conflicts are listed,\n    instead.  (This is useful for editing all files with text conflicts.)\n\n    Use brz resolve when you have fixed a problem.\n    '
    takes_options = ['directory', option.Option('text', help='List paths of files with text conflicts.')]
    _see_also = ['resolve', 'conflict-types']

    def run(self, text=False, directory='.'):
        wt = workingtree.WorkingTree.open_containing(directory)[0]
        for conflict in wt.conflicts():
            if text:
                if conflict.typestring != 'text conflict':
                    continue
                self.outf.write(conflict.path + '\n')
            else:
                self.outf.write(str(conflict) + '\n')