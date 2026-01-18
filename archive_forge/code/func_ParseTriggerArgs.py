from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseTriggerArgs(args, messages):
    """Parses flags generic to all triggers.

  Args:
    args: An argparse arguments object.
    messages: A Cloud Build messages module

  Returns:
    A partially populated build trigger and a boolean indicating whether or not
    the full trigger was loaded from a file.
  """
    if args.trigger_config:
        trigger = cloudbuild_util.LoadMessageFromPath(path=args.trigger_config, msg_type=messages.BuildTrigger, msg_friendly_name='build trigger config', skip_camel_case=['substitutions'])
        return (trigger, True)
    trigger = messages.BuildTrigger()
    trigger.name = args.name
    trigger.description = args.description
    trigger.serviceAccount = args.service_account
    ParseRequireApproval(trigger, args, messages)
    return (trigger, False)