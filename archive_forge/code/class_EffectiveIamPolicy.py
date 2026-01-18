from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveIamPolicy(_messages.Message):
    """The effective IAM policies on one resource.

  Fields:
    fullResourceName: The [full_resource_name]
      (https://cloud.google.com/asset-inventory/docs/resource-name-format) for
      which the policies are computed. This is one of the
      BatchGetEffectiveIamPoliciesRequest.names the caller provides in the
      request.
    policies: The effective policies for the full_resource_name. These
      policies include the policy set on the full_resource_name and those set
      on its parents and ancestors up to the
      BatchGetEffectiveIamPoliciesRequest.scope. Note that these policies are
      not filtered according to the resource type of the full_resource_name.
      These policies are hierarchically ordered by
      PolicyInfo.attached_resource starting from full_resource_name itself to
      its parents and ancestors, such that policies[i]'s
      PolicyInfo.attached_resource is the child of policies[i+1]'s
      PolicyInfo.attached_resource, if policies[i+1] exists.
  """
    fullResourceName = _messages.StringField(1)
    policies = _messages.MessageField('PolicyInfo', 2, repeated=True)