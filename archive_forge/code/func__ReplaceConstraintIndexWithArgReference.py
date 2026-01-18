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
def _ReplaceConstraintIndexWithArgReference(arguments, positionals):
    for i, arg in enumerate(arguments):
        if isinstance(arg, int):
            if arg < 0:
                arguments[i] = positionals[-(arg + 1)]
            else:
                arguments[i] = all_flags_list[arg]
        elif arg.get(LOOKUP_IS_GROUP, False):
            _ReplaceConstraintIndexWithArgReference(arg.get(LOOKUP_ARGUMENTS), positionals)