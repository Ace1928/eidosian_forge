from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def SetInitialOwner(unused_ref, args, request):
    """Set the initial owner.

  Defaults to 'empty' for dynamic groups and to 'with-initial-owner' for
  other group types.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    if args.IsSpecified('with_initial_owner'):
        return request
    version = GetApiVersion(args)
    messages = ci_client.GetMessages(version)
    create_message = messages.CloudidentityGroupsCreateRequest
    config_enum = create_message.InitialGroupConfigValueValuesEnum
    if args.IsSpecified('group_type') and 'dynamic' in args.group_type or (args.IsSpecified('labels') and 'dynamic' in args.labels):
        request.initialGroupConfig = config_enum.EMPTY
    else:
        request.initialGroupConfig = config_enum.WITH_INITIAL_OWNER
    return request