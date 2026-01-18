from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CorsSettings(_messages.Message):
    """Allows customers to configure HTTP request paths that'll allow HTTP
  OPTIONS call to bypass authentication and authorization.

  Fields:
    allowHttpOptions: Configuration to allow HTTP OPTIONS calls to skip
      authorization. If undefined, IAP will not apply any special logic to
      OPTIONS requests.
  """
    allowHttpOptions = _messages.BooleanField(1)