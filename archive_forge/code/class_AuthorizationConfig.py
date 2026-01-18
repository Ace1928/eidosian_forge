from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationConfig(_messages.Message):
    """Configuration of authorization.  This section determines the
  authorization provider, if unspecified, then no authorization check will be
  done.  Example:      experimental:       authorization:         provider:
  firebaserules.googleapis.com

  Fields:
    provider: The name of the authorization provider, such as
      firebaserules.googleapis.com.
  """
    provider = _messages.StringField(1)