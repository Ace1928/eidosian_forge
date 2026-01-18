from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsTrustConfigsDeleteRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsTrustConfigsDeleteRequest object.

  Fields:
    etag: The current etag of the TrustConfig. If an etag is provided and does
      not match the current etag of the resource, deletion will be blocked and
      an ABORTED error will be returned.
    name: Required. A name of the TrustConfig to delete. Must be in the format
      `projects/*/locations/*/trustConfigs/*`.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)