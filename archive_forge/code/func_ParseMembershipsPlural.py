from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def ParseMembershipsPlural(args, prompt=False, prompt_cancel=True, autoselect=False, allow_cross_project=False, search=False):
    """Parses a list of membership resources from args.

  Allows for a `--memberships` flag and a `--all-memberships` flag.

  Args:
    args: object containing arguments passed as flags with the command
    prompt: whether to prompt in console for a membership when none are provided
      in args
    prompt_cancel: whether to include a 'cancel' option in the prompt
    autoselect: if no memberships are provided and only one exists,
      automatically use that one
    allow_cross_project: whether to allow memberships from different projects
    search: whether to check that the membership exists in the fleet

  Returns:
    memberships: A list of membership resource name strings

  Raises:
    exceptions.Error if no memberships were found or memberships are invalid
    calliope_exceptions.RequiredArgumentException if membership was not provided
  """
    memberships = []
    if hasattr(args, 'all_memberships') and args.all_memberships:
        all_memberships, unreachable = api_util.ListMembershipsFull(filter_cluster_missing=True)
        if unreachable:
            raise exceptions.Error('Locations {} are currently unreachable. Please try again or specify memberships for this command.'.format(unreachable))
        if not all_memberships:
            raise exceptions.Error('No Memberships available in the fleet.')
        return all_memberships
    if args.IsKnownAndSpecified('memberships'):
        if resources.MembershipLocationSpecified(args):
            memberships += resources.PluralMembershipsResourceNames(args)
            if search:
                for membership in memberships:
                    if not api_util.GetMembership(membership):
                        raise exceptions.Error('Membership {} does not exist in the fleet.'.format(membership))
            if not allow_cross_project and len(resources.GetMembershipProjects(memberships)) > 1:
                raise CrossProjectError(resources.GetMembershipProjects(memberships))
        else:
            memberships += resources.SearchMembershipResourcesPlural(args, filter_cluster_missing=True)
    if memberships:
        return memberships
    if not prompt and (not autoselect):
        raise MembershipRequiredError(args)
    all_memberships, unreachable = api_util.ListMembershipsFull(filter_cluster_missing=True)
    if unreachable:
        raise exceptions.Error('Locations {} are currently unreachable. Please specify memberships using `--location` or the full resource name (projects/*/locations/*/memberships/*)'.format(unreachable))
    if autoselect and len(all_memberships) == 1:
        log.status.Print('Selecting membership [{}].'.format(all_memberships[0]))
        return [all_memberships[0]]
    if prompt:
        membership = resources.PromptForMembership(cancel=prompt_cancel)
        if membership:
            memberships.append(membership)
        return memberships
    raise MembershipRequiredError(args)