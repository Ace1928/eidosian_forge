from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentitySelector(_messages.Message):
    """Specifies an identity for which to determine resource access, based on
  roles assigned either directly to them or to the groups they belong to,
  directly or indirectly.

  Fields:
    identity: Required. The identity appear in the form of principals in [IAM
      policy binding](https://cloud.google.com/iam/reference/rest/v1/Binding).
      The examples of supported forms are: "user:mike@example.com",
      "group:admins@example.com", "domain:google.com", "serviceAccount:my-
      project-id@appspot.gserviceaccount.com". Notice that wildcard characters
      (such as * and ?) are not supported. You must give a specific identity.
  """
    identity = _messages.StringField(1)