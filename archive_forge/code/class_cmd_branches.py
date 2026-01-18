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
class cmd_branches(Command):
    __doc__ = 'List the branches available at the current location.\n\n    This command will print the names of all the branches at the current\n    location.\n    '
    takes_args = ['location?']
    takes_options = [Option('recursive', short_name='R', help='Recursively scan for branches rather than just looking in the specified location.')]

    def run(self, location='.', recursive=False):
        if recursive:
            t = transport.get_transport(location, purpose='read')
            if not t.listable():
                raise errors.CommandError("Can't scan this type of location.")
            for b in controldir.ControlDir.find_branches(t):
                self.outf.write('%s\n' % urlutils.unescape_for_display(urlutils.relative_url(t.base, b.base), self.outf.encoding).rstrip('/'))
        else:
            dir = controldir.ControlDir.open_containing(location)[0]
            try:
                active_branch = dir.open_branch(name='')
            except errors.NotBranchError:
                active_branch = None
            names = {}
            for name, branch in iter_sibling_branches(dir):
                if name == '':
                    continue
                active = active_branch is not None and active_branch.user_url == branch.user_url
                names[name] = active
            if not any(names.values()) and active_branch is not None:
                self.outf.write('* %s\n' % gettext('(default)'))
            for name in sorted(names):
                active = names[name]
                if active:
                    prefix = '*'
                else:
                    prefix = ' '
                self.outf.write('{} {}\n'.format(prefix, name))