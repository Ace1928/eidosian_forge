from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def SetTransitiveQuery(unused_ref, args, request):
    """Sets query paremeters to request.query.

  Args:
    unused_ref: unused.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    params = []
    if hasattr(args, 'member_email') and args.IsSpecified('member_email'):
        params.append("member_key_id=='{}'".format(args.member_email))
    if hasattr(args, 'labels') and args.IsSpecified('labels'):
        params.append("'{}' in labels".format(args.labels))
    request.query = '&&'.join(params)
    return request