from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsTlsInspectionPoliciesPatchRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsTlsInspectionPoliciesPatchRequest
  object.

  Fields:
    name: Required. Name of the resource. Name is of the form projects/{projec
      t}/locations/{location}/tlsInspectionPolicies/{tls_inspection_policy}
      tls_inspection_policy should match the
      pattern:(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    tlsInspectionPolicy: A TlsInspectionPolicy resource to be passed as the
      request body.
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the TlsInspectionPolicy resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all fields will be overwritten.
  """
    name = _messages.StringField(1, required=True)
    tlsInspectionPolicy = _messages.MessageField('TlsInspectionPolicy', 2)
    updateMask = _messages.StringField(3)