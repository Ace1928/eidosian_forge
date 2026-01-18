from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _ReplaceFlagWithIndex(command):
    for name, flag in six.iteritems(command.flags):
        command.flags[name] = all_flags[_FlagIndexKey(flag)].index
        _ReplaceConstraintFlagWithIndex(command.constraints.arguments)
    for subcommand in command.commands.values():
        _ReplaceFlagWithIndex(subcommand)