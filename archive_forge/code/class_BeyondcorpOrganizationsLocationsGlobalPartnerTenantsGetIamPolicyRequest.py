from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetIamPolicyRequest(_messages.Message):
    """A
  BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetIamPolicyRequest
  object.

  Fields:
    options_requestedPolicyVersion: Optional. The maximum policy version that
      will be used to format the policy. Valid values are 0, 1, and 3.
      Requests specifying an invalid value will be rejected. Requests for
      policies with any conditional role bindings must specify version 3.
      Policies with no conditional role bindings may specify any valid value
      or leave the field unset. The policy in the response might use the
      policy version that you specified, or it might use a lower policy
      version. For example, if you specify version 3, but the policy has no
      conditional role bindings, the response uses version 1. To learn which
      resources support conditions in their IAM policies, see the [IAM
      documentation](https://cloud.google.com/iam/help/conditions/resource-
      policies).
    resource: REQUIRED: The resource for which the policy is being requested.
      See [Resource
      names](https://cloud.google.com/apis/design/resource_names) for the
      appropriate value for this field.
  """
    options_requestedPolicyVersion = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    resource = _messages.StringField(2, required=True)