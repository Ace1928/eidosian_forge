from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def InstanceActionChoicesWithNone(flag_prefix=''):
    """Return possible instance action choices with NONE value."""
    return _CombineOrderedChoices({'none': 'No action'}, InstanceActionChoicesWithoutNone(flag_prefix))