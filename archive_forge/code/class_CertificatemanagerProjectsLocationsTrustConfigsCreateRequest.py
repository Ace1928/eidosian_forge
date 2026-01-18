from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsTrustConfigsCreateRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsTrustConfigsCreateRequest object.

  Fields:
    parent: Required. The parent resource of the TrustConfig. Must be in the
      format `projects/*/locations/*`.
    trustConfig: A TrustConfig resource to be passed as the request body.
    trustConfigId: Required. A user-provided name of the TrustConfig. Must
      match the regexp `[a-z0-9-]{1,63}`.
  """
    parent = _messages.StringField(1, required=True)
    trustConfig = _messages.MessageField('TrustConfig', 2)
    trustConfigId = _messages.StringField(3)