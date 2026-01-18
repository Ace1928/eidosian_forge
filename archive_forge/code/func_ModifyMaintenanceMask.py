from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib import memcache
def ModifyMaintenanceMask(unused_ref, args, req):
    """Update patch mask for maintenancePolicy.

  Args:
    unused_ref: The field resource reference.
    args: The parsed arg namespace.
    req: The auto-generated patch request.
  Returns:
    FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest
  """
    policy_is_updated = hasattr(req, 'instance') and hasattr(req.instance, 'maintenancePolicy') and req.instance.maintenancePolicy
    if args.IsSpecified('maintenance_window_any') or policy_is_updated:
        policy = 'maintenancePolicy'
        mask = list(filter(lambda m: m and policy not in m, req.updateMask.split(',')))
        AddFieldToUpdateMask(mask, policy)
        req.updateMask = ','.join(mask)
    return req