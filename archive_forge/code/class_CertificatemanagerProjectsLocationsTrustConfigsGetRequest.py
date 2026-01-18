from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsTrustConfigsGetRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsTrustConfigsGetRequest object.

  Fields:
    name: Required. A name of the TrustConfig to describe. Must be in the
      format `projects/*/locations/*/trustConfigs/*`.
  """
    name = _messages.StringField(1, required=True)