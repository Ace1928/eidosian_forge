from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationCode(_messages.Message):
    """Defines an authorization code.

  Fields:
    code: The Authorization Code in ASCII. It can be used to transfer the
      domain to or from another registrar.
  """
    code = _messages.StringField(1)