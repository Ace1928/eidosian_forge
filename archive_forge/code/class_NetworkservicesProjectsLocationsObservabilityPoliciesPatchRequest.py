from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsObservabilityPoliciesPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsObservabilityPoliciesPatchRequest
  object.

  Fields:
    name: Required. Name of the ObservabilityPolicy resource. It matches
      pattern `projects/*/locations/global/observabilityPolicies/`.
    observabilityPolicy: A ObservabilityPolicy resource to be passed as the
      request body.
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the ObservabilityPolicy resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all fields will be overwritten.
  """
    name = _messages.StringField(1, required=True)
    observabilityPolicy = _messages.MessageField('ObservabilityPolicy', 2)
    updateMask = _messages.StringField(3)