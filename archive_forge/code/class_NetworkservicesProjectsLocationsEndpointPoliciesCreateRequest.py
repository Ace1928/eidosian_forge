from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEndpointPoliciesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEndpointPoliciesCreateRequest object.

  Fields:
    endpointPolicy: A EndpointPolicy resource to be passed as the request
      body.
    endpointPolicyId: Required. Short name of the EndpointPolicy resource to
      be created. E.g. "CustomECS".
    parent: Required. The parent resource of the EndpointPolicy. Must be in
      the format `projects/*/locations/global`.
  """
    endpointPolicy = _messages.MessageField('EndpointPolicy', 1)
    endpointPolicyId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)