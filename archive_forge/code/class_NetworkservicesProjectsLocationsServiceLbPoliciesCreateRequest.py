from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsServiceLbPoliciesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsServiceLbPoliciesCreateRequest object.

  Fields:
    parent: Required. The parent resource of the ServiceLbPolicy. Must be in
      the format `projects/{project}/locations/{location}`.
    serviceLbPolicy: A ServiceLbPolicy resource to be passed as the request
      body.
    serviceLbPolicyId: Required. Short name of the ServiceLbPolicy resource to
      be created. E.g. for resource name `projects/{project}/locations/{locati
      on}/serviceLbPolicies/{service_lb_policy_name}`. the id is value of
      {service_lb_policy_name}
  """
    parent = _messages.StringField(1, required=True)
    serviceLbPolicy = _messages.MessageField('ServiceLbPolicy', 2)
    serviceLbPolicyId = _messages.StringField(3)