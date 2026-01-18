import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def _show_help(self, parser, global_options=1, display_options=1, commands=[]):
    """Show help for the setup script command-line in the form of
        several lists of command-line options.  'parser' should be a
        FancyGetopt instance; do not expect it to be returned in the
        same state, as its option table will be reset to make it
        generate the correct help text.

        If 'global_options' is true, lists the global options:
        --verbose, --dry-run, etc.  If 'display_options' is true, lists
        the "display-only" options: --name, --version, etc.  Finally,
        lists per-command help for every command name or command class
        in 'commands'.
        """
    from distutils.core import gen_usage
    from distutils.cmd import Command
    if global_options:
        if display_options:
            options = self._get_toplevel_options()
        else:
            options = self.global_options
        parser.set_option_table(options)
        parser.print_help(self.common_usage + '\nGlobal options:')
        print('')
    if display_options:
        parser.set_option_table(self.display_options)
        parser.print_help('Information display options (just display ' + 'information, ignore any commands)')
        print('')
    for command in self.commands:
        if isinstance(command, type) and issubclass(command, Command):
            klass = command
        else:
            klass = self.get_command_class(command)
        if hasattr(klass, 'help_options') and isinstance(klass.help_options, list):
            parser.set_option_table(klass.user_options + fix_help_options(klass.help_options))
        else:
            parser.set_option_table(klass.user_options)
        parser.print_help("Options for '%s' command:" % klass.__name__)
        print('')
    print(gen_usage(self.script_name))