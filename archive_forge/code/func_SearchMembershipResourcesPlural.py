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
def SearchMembershipResourcesPlural(args, filter_cluster_missing=False):
    """Searches the fleet for the membership resources provided in args.

  Only necessary if location is ambiguous, i.e.
  MembershipLocationSpecified(args) is
  False. Assumes the argument is called `--membership`, `--memberships` if
  plural, or `MEMBERSHIP_NAME` if positional. Runs ListMemberships API call to
  verify the membership exists.

  Args:
    args: arguments provided to a command, including a membership resource arg
    filter_cluster_missing: whether to filter out memberships that are missing
    a cluster.

  Returns:
    A list of membership resource names
      (e.g. ["projects/x/locations/y/memberships/z"])

  Raises:
    googlecloudsdk.core.exceptions.Error: unable to find a membership
      in the fleet
  """
    if args.IsKnownAndSpecified('memberships'):
        arg_memberships = args.memberships
    else:
        return None
    all_memberships, unavailable = api_util.ListMembershipsFull(filter_cluster_missing=filter_cluster_missing)
    if unavailable:
        raise exceptions.Error('Locations [{}] are currently unreachable. Please specify memberships using `--location` or the full resource name (projects/*/locations/*/memberships/*)'.format(unavailable))
    if not all_memberships:
        raise exceptions.Error('No memberships available in the fleet.')
    memberships = []
    for arg_membership in arg_memberships:
        found = []
        for existing_membership in all_memberships:
            if arg_membership == util.MembershipShortname(existing_membership):
                found.append(existing_membership)
        if not found:
            raise exceptions.Error('Membership {} not found in the fleet.'.format(arg_membership))
        elif len(found) > 1:
            raise AmbiguousMembershipError(arg_membership)
        memberships.append(found[0])
    return memberships