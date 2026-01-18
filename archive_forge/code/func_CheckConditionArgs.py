from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def CheckConditionArgs(args):
    """Checks if condition arguments exist and are specified correctly.

  Args:
    args: argparse.Namespace, the parsed arguments.
  Returns:
    bool: True, if '--condition-filter' is specified.
  Raises:
    RequiredArgumentException:
      if '--if' is not set but '--condition-filter' is specified.
    InvalidArgumentException:
      if flag in should_not_be_set is specified without '--condition-filter'.
  """
    if args.IsSpecified('condition_filter'):
        if not args.IsSpecified('if_value'):
            raise calliope_exc.RequiredArgumentException('--if', 'If --condition-filter is set then --if must be set as well.')
        return True
    else:
        should_not_be_set = ['--aggregation', '--duration', '--trigger-count', '--trigger-percent', '--condition-display-name', '--if', '--combiner']
        for flag in should_not_be_set:
            if flag == '--if':
                dest = 'if_value'
            else:
                dest = _FlagToDest(flag)
            if args.IsSpecified(dest):
                raise calliope_exc.InvalidArgumentException(flag, 'Should only be specified if --condition-filter is also specified.')
        return False