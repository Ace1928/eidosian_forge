from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationRule(_messages.Message):
    """Authorization rule for API services.  It specifies the permission(s)
  required for an API element for the overall API request to succeed. It is
  typically used to mark request message fields that contain the name of the
  resource and indicates the permissions that will be checked on that
  resource.  For example:      package google.storage.v1;      message
  CopyObjectRequest {       string source = 1 [
  (google.api.authz).permissions = "storage.objects.get"];        string
  destination = 2 [         (google.api.authz).permissions =
  "storage.objects.create,storage.objects.update"];     }

  Fields:
    permissions: The required permissions. The acceptable values vary depend
      on the authorization system used. For Google APIs, it should be a comma-
      separated Google IAM permission values. When multiple permissions are
      listed, the semantics is not defined by the system. Additional
      documentation must be provided manually.
    selector: Selects the API elements to which this rule applies.  Refer to
      selector for syntax details.
  """
    permissions = _messages.StringField(1)
    selector = _messages.StringField(2)