from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.organizations import org_utils
def SetOrganization(unused_ref, args, request):
    """Set organization ID to request.organizationId.

  Args:
    unused_ref: A string representing the operation reference. Unused and may
      be None.
    args: The argparse namespace.
    request: The request to modify.

  Returns:
    The updated request.
  """
    org_id = org_utils.GetOrganizationId(args.organization)
    if org_id:
        request.organizationsId = org_id
        return request
    else:
        raise org_utils.UnknownOrganizationError(args.organization)