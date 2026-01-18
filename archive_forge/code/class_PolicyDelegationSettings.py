from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyDelegationSettings(_messages.Message):
    """PolicyDelegationConfig allows google-internal teams to use IAP for apps
  hosted in a tenant project. Using these settings, the app can delegate
  permission check to happen against the linked customer project. This is only
  ever supposed to be used by google internal teams, hence the restriction on
  the proto.

  Fields:
    iamPermission: Permission to check in IAM.
    iamServiceName: The DNS name of the service (e.g.
      "resourcemanager.googleapis.com"). This should be the domain name part
      of the full resource names (see https://aip.dev/122#full-resource-
      names), which is usually the same as IamServiceSpec.service of the
      service where the resource type is defined.
    policyName: Policy name to be checked
    resource: IAM resource to check permission on
  """
    iamPermission = _messages.StringField(1)
    iamServiceName = _messages.StringField(2)
    policyName = _messages.MessageField('PolicyName', 3)
    resource = _messages.MessageField('Resource', 4)