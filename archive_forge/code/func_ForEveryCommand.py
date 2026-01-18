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
def ForEveryCommand(self, command):
    if command.cli_name not in self._allowlist:
        self._issues.append(LintError(name=self.name, command=command, error_message='command name [{0}] is not allowlisted'.format(command.cli_name)))