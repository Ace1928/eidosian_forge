from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FirewallActionSubstituteAction(_messages.Message):
    """A substitute action transparently serves a different page than the one
  requested.

  Fields:
    path: Optional. The address to redirect to. The target is a relative path
      in the current host. Example: "/blog/404.html".
  """
    path = _messages.StringField(1)