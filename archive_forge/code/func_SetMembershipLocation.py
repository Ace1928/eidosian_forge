from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.memberships import errors
def SetMembershipLocation(ref, args, request):
    """Set membership location for requested resource.

  Args:
    ref: reference to the membership object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref
    if args.IsKnownAndSpecified('membership'):
        if resources.MembershipLocationSpecified(args):
            request.name = resources.MembershipResourceName(args)
        else:
            request.name = resources.SearchMembershipResource(args)
    else:
        raise calliope_exceptions.RequiredArgumentException('MEMBERSHIP', 'membership is required for this command.')
    return request