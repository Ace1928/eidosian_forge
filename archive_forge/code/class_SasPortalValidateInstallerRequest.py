from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalValidateInstallerRequest(_messages.Message):
    """Request for ValidateInstaller.

  Fields:
    encodedSecret: Required. JSON Web Token signed using a CPI private key.
      Payload must include a "secret" claim whose value is the secret.
    installerId: Required. Unique installer id (CPI ID) from the Certified
      Professional Installers database.
    secret: Required. Secret returned by the GenerateSecret.
  """
    encodedSecret = _messages.StringField(1)
    installerId = _messages.StringField(2)
    secret = _messages.StringField(3)