from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsAuthorizedCertificatesPatchRequest(_messages.Message):
    """A AppengineAppsAuthorizedCertificatesPatchRequest object.

  Fields:
    authorizedCertificate: A AuthorizedCertificate resource to be passed as
      the request body.
    name: Name of the resource to update. Example:
      apps/myapp/authorizedCertificates/12345.
    updateMask: Standard field mask for the set of fields to be updated.
      Updates are only supported on the certificate_raw_data and display_name
      fields.
  """
    authorizedCertificate = _messages.MessageField('AuthorizedCertificate', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)