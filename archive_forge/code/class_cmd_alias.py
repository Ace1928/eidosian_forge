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
class cmd_alias(Command):
    __doc__ = 'Set/unset and display aliases.\n\n    :Examples:\n        Show the current aliases::\n\n            brz alias\n\n        Show the alias specified for \'ll\'::\n\n            brz alias ll\n\n        Set an alias for \'ll\'::\n\n            brz alias ll="log --line -r-10..-1"\n\n        To remove an alias for \'ll\'::\n\n            brz alias --remove ll\n\n    '
    takes_args = ['name?']
    takes_options = [Option('remove', help='Remove the alias.')]

    def run(self, name=None, remove=False):
        if remove:
            self.remove_alias(name)
        elif name is None:
            self.print_aliases()
        else:
            equal_pos = name.find('=')
            if equal_pos == -1:
                self.print_alias(name)
            else:
                self.set_alias(name[:equal_pos], name[equal_pos + 1:])

    def remove_alias(self, alias_name):
        if alias_name is None:
            raise errors.CommandError(gettext('brz alias --remove expects an alias to remove.'))
        c = _mod_config.GlobalConfig()
        c.unset_alias(alias_name)

    @display_command
    def print_aliases(self):
        """Print out the defined aliases in a similar format to bash."""
        aliases = _mod_config.GlobalConfig().get_aliases()
        for key, value in sorted(aliases.items()):
            self.outf.write('brz alias {}="{}"\n'.format(key, value))

    @display_command
    def print_alias(self, alias_name):
        from .commands import get_alias
        alias = get_alias(alias_name)
        if alias is None:
            self.outf.write('brz alias: %s: not found\n' % alias_name)
        else:
            self.outf.write('brz alias {}="{}"\n'.format(alias_name, ' '.join(alias)))

    def set_alias(self, alias_name, alias_command):
        """Save the alias in the global config."""
        c = _mod_config.GlobalConfig()
        c.set_alias(alias_name, alias_command)