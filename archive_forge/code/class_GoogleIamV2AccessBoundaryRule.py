from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2AccessBoundaryRule(_messages.Message):
    """An IAM access boundary rule, which defines an upper bound of IAM
  permissions on a single resource. All access boundary rules in an access
  boundary policy are evaluated together as a union. Even if this access
  boundary rule does not allow access to the resource, another access boundary
  rule might allow access.

  Fields:
    availabilityCondition: Optional. An availability condition that further
      constrains the access allowed by the access boundary rule. If the
      condition evaluates to `true`, then this access boundary rule will
      provide access to the specified resource, assuming the principal has the
      required permissions for the resource. If the condition does not
      evaluate to `true`, then access to the specified resource will not be
      available. The condition can only evaluate the access level for the
      request. Access levels use the format
      `accessPolicies/{policy_name}/accessLevels/{access_level_shortname}`.
    availablePermissions: Required. A list of permissions that may be allowed
      for use on the specified resource. The only supported value is `*`,
      which represents all permissions.
    availableResource: Required. The full resource name of a Google Cloud
      resource. The format is defined at
      https://cloud.google.com/apis/design/resource_names. The only supported
      value is `*`, which represents all resources.
  """
    availabilityCondition = _messages.MessageField('GoogleTypeExpr', 1)
    availablePermissions = _messages.StringField(2, repeated=True)
    availableResource = _messages.StringField(3)