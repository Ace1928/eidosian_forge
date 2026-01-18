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
def AddMinimalActionArg(parser, choices_with_none=True, default=None):
    choices = InstanceActionChoicesWithNone() if choices_with_none else InstanceActionChoicesWithoutNone()
    parser.add_argument('--minimal-action', choices=choices, default=default, help="Use this flag to minimize disruption as much as possible or to\n        apply a more disruptive action than is strictly necessary.\n        The MIG performs at least this action on each instance while\n        updating. If the update requires a more disruptive action than\n        the one specified here, then the more disruptive action is\n        performed. If you omit this flag, the update uses the\n        ``minimal-action'' value from the MIG's update policy, unless it\n        is not set in which case the default is ``replace''.")