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
def AddMostDisruptiveActionArg(parser, choices_with_none=True, default=None):
    choices = InstanceActionChoicesWithNone() if choices_with_none else InstanceActionChoicesWithoutNone()
    parser.add_argument('--most-disruptive-allowed-action', choices=choices, default=default, help="Use this flag to prevent an update if it requires more disruption\n        than you can afford. At most, the MIG performs the specified\n        action on each instance while updating. If the update requires\n        a more disruptive action than the one specified here, then\n        the update fails and no changes are made. If you omit this flag,\n        the update uses the ``most-disruptive-allowed-action'' value from\n        the MIG's update policy, unless it is not set in which case\n        the default is ``replace''.")