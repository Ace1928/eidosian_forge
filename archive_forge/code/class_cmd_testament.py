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
class cmd_testament(Command):
    __doc__ = 'Show testament (signing-form) of a revision.'
    takes_options = ['revision', Option('long', help='Produce long-format testament.'), Option('strict', help='Produce a strict-format testament.')]
    takes_args = ['branch?']
    encoding_type = 'exact'

    @display_command
    def run(self, branch='.', revision=None, long=False, strict=False):
        from .bzr.testament import StrictTestament, Testament
        if strict is True:
            testament_class = StrictTestament
        else:
            testament_class = Testament
        if branch == '.':
            b = Branch.open_containing(branch)[0]
        else:
            b = Branch.open(branch)
        self.enter_context(b.lock_read())
        if revision is None:
            rev_id = b.last_revision()
        else:
            rev_id = revision[0].as_revision_id(b)
        t = testament_class.from_revision(b.repository, rev_id)
        if long:
            self.outf.writelines(t.as_text_lines())
        else:
            self.outf.write(t.as_short_text())