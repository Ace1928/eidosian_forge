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
class cmd_whoami(Command):
    __doc__ = 'Show or set brz user id.\n\n    :Examples:\n        Show the email of the current user::\n\n            brz whoami --email\n\n        Set the current user::\n\n            brz whoami "Frank Chu <fchu@example.com>"\n    '
    takes_options = ['directory', Option('email', help='Display email address only.'), Option('branch', help='Set identity for the current branch instead of globally.')]
    takes_args = ['name?']
    encoding_type = 'replace'

    @display_command
    def run(self, email=False, branch=False, name=None, directory=None):
        if name is None:
            if directory is None:
                try:
                    c = Branch.open_containing('.')[0].get_config_stack()
                except errors.NotBranchError:
                    c = _mod_config.GlobalStack()
            else:
                c = Branch.open(directory).get_config_stack()
            identity = c.get('email')
            if email:
                self.outf.write(_mod_config.extract_email_address(identity) + '\n')
            else:
                self.outf.write(identity + '\n')
            return
        if email:
            raise errors.CommandError(gettext('--email can only be used to display existing identity'))
        try:
            _mod_config.extract_email_address(name)
        except _mod_config.NoEmailInUsername:
            warning('"%s" does not seem to contain an email address.  This is allowed, but not recommended.', name)
        if branch:
            if directory is None:
                c = Branch.open_containing('.')[0].get_config_stack()
            else:
                b = Branch.open(directory)
                self.enter_context(b.lock_write())
                c = b.get_config_stack()
        else:
            c = _mod_config.GlobalStack()
        c.set('email', name)