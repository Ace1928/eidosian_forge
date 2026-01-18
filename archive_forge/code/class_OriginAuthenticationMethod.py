from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginAuthenticationMethod(_messages.Message):
    """[Deprecated] Configuration for the origin authentication method.
  Configuration for the origin authentication method.

  Fields:
    jwt: A Jwt attribute.
  """
    jwt = _messages.MessageField('Jwt', 1)