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
def GetBasePolicyMessageFromArgs(args, policy_class):
    """Returns the base policy from args."""
    if args.IsSpecified('policy') or args.IsSpecified('policy_from_file'):
        policy_string = args.policy or args.policy_from_file
        policy = MessageFromString(policy_string, policy_class, 'AlertPolicy')
    else:
        policy = policy_class()
    return policy