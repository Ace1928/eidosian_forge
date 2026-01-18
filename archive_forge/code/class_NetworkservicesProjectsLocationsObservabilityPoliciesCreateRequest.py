from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsObservabilityPoliciesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsObservabilityPoliciesCreateRequest
  object.

  Fields:
    observabilityPolicy: A ObservabilityPolicy resource to be passed as the
      request body.
    observabilityPolicyId: Required. Short name of the ObservabilityPolicy
      resource to be created. E.g. TODO(Add an example).
    parent: Required. The parent resource of the ObservabilityPolicy. Must be
      in the format `projects/*/locations/global`.
  """
    observabilityPolicy = _messages.MessageField('ObservabilityPolicy', 1)
    observabilityPolicyId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)