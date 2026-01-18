from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
class NameChecker(Checker):
    """Checks if group,command and flags names have underscores or mixed case."""
    name = 'NameCheck'
    description = 'Verifies all existing flags not to have underscores.'

    def __init__(self):
        super(NameChecker, self).__init__()
        self._issues = []

    def _ForEvery(self, cmd_or_group):
        """Run name check for given command or group."""
        if '_' in cmd_or_group.cli_name:
            self._issues.append(LintError(name=NameChecker.name, command=cmd_or_group, error_message='command name [{0}] has underscores'.format(cmd_or_group.cli_name)))
        if not (cmd_or_group.cli_name.islower() or cmd_or_group.cli_name.isupper()):
            self._issues.append(LintError(name=NameChecker.name, command=cmd_or_group, error_message='command name [{0}] mixed case'.format(cmd_or_group.cli_name)))
        for flag in cmd_or_group.GetSpecificFlags():
            if not any((f.startswith('--') for f in flag.option_strings)):
                if len(flag.option_strings) != 1 or flag.option_strings[0] != '-h':
                    self._issues.append(LintError(name=NameChecker.name, command=cmd_or_group, error_message='flag [{0}] has no long form'.format(','.join(flag.option_strings))))
            for flag_option_string in flag.option_strings:
                msg = None
                if '_' in flag_option_string:
                    msg = 'flag [%s] has underscores' % flag_option_string
                if flag_option_string.startswith('--') and (not flag_option_string.islower()):
                    msg = 'long flag [%s] has upper case characters' % flag_option_string
                if msg:
                    self._issues.append(LintError(name=NameChecker.name, command=cmd_or_group, error_message=msg))

    def ForEveryGroup(self, group):
        self._ForEvery(group)

    def ForEveryCommand(self, command):
        self._ForEvery(command)

    def End(self):
        return self._issues