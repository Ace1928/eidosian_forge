from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.memberships import errors
def ExecuteUpdateMembershipRequest(ref, args):
    """Set membership location for requested resource.

  Args:
    ref: API response from update membership call
    args: command line arguments.

  Returns:
    response
  """
    del ref
    if resources.MembershipLocationSpecified(args):
        name = resources.MembershipResourceName(args)
    else:
        name = resources.SearchMembershipResource(args)
    release_track = args.calliope_command.ReleaseTrack()
    obj = api_util.GetMembership(name, release_track)
    update_fields = []
    description = external_id = infra_type = None
    if release_track == calliope_base.ReleaseTrack.BETA and args.GetValue('description'):
        update_fields.append('description')
        description = args.GetValue('description')
    if args.GetValue('external_id'):
        update_fields.append('externalId')
        external_id = args.GetValue('external_id')
    if release_track != calliope_base.ReleaseTrack.GA and args.GetValue('infra_type'):
        update_fields.append('infrastructureType')
        infra_type = args.GetValue('infra_type')
    if args.GetValue('clear_labels') or args.GetValue('update_labels') or args.GetValue('remove_labels'):
        update_fields.append('labels')
    update_mask = ','.join(update_fields)
    response = api_util.UpdateMembership(name, obj, update_mask, release_track, description=description, external_id=external_id, infra_type=infra_type, clear_labels=args.GetValue('clear_labels'), update_labels=args.GetValue('update_labels'), remove_labels=args.GetValue('remove_labels'), issuer_url=None, oidc_jwks=None, api_server_version=None, async_flag=args.GetValue('async'))
    return response