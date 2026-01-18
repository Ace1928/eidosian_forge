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
def _ReplaceIndexWithFlagReference(command):
    flags = command[LOOKUP_FLAGS]
    for name, index in six.iteritems(flags):
        flags[name] = all_flags_list[index]
    arguments = command[LOOKUP_CONSTRAINTS][LOOKUP_ARGUMENTS]
    _ReplaceConstraintIndexWithArgReference(arguments, command[LOOKUP_POSITIONALS])
    for subcommand in command[LOOKUP_COMMANDS].values():
        _ReplaceIndexWithFlagReference(subcommand)