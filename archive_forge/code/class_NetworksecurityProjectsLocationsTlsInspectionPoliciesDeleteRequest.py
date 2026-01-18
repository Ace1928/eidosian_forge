from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsTlsInspectionPoliciesDeleteRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsTlsInspectionPoliciesDeleteRequest
  object.

  Fields:
    force: If set to true, any rules for this TlsInspectionPolicy will also be
      deleted. (Otherwise, the request will only work if the
      TlsInspectionPolicy has no rules.)
    name: Required. A name of the TlsInspectionPolicy to delete. Must be in
      the format `projects/{project}/locations/{location}/tlsInspectionPolicie
      s/{tls_inspection_policy}`.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)