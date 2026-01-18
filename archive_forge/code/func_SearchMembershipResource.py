from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def SearchMembershipResource(args, flag_override='', filter_cluster_missing=False):
    """Searches the fleet for an ambiguous membership provided in args.

  Only necessary if location is ambiguous, i.e.
  MembershipLocationSpecified(args) is False, or this behavior is necessary for
  backwards compatibility. If flag_override is unset, the argument must be
  called `MEMBERSHIP_NAME` if positional and  `--membership` otherwise. Runs a
  ListMemberships API call to verify the membership exists.

  Args:
    args: arguments provided to a command, including a membership resource arg
    flag_override: a custom membership flag
    filter_cluster_missing: whether to filter out memberships that are missing
    a cluster.

  Returns:
    A membership resource name string
      (e.g. projects/x/locations/y/memberships/z)

  Raises:
    googlecloudsdk.core.exceptions.Error: unable to find the membership
      in the fleet
  """
    if MembershipLocationSpecified(args) and api_util.GetMembership(MembershipResourceName(args)):
        return MembershipResourceName(args)
    if args.IsKnownAndSpecified(flag_override):
        arg_membership = getattr(args, flag_override)
    elif args.IsKnownAndSpecified('MEMBERSHIP_NAME'):
        arg_membership = args.MEMBERSHIP_NAME
    elif args.IsKnownAndSpecified('membership'):
        arg_membership = args.membership
    else:
        return None
    all_memberships, unavailable = api_util.ListMembershipsFull(filter_cluster_missing=filter_cluster_missing)
    if unavailable:
        raise exceptions.Error('Locations {} are currently unreachable. Please specify memberships using `--location` or the full resource name (projects/*/locations/*/memberships/*)'.format(unavailable))
    if not all_memberships:
        raise exceptions.Error('No memberships available in the fleet.')
    found = []
    for existing_membership in all_memberships:
        if arg_membership == util.MembershipShortname(existing_membership):
            found.append(existing_membership)
    if not found:
        raise exceptions.Error('Membership {} not found in the fleet.'.format(arg_membership))
    elif len(found) > 1:
        raise AmbiguousMembershipError(arg_membership)
    return found[0]