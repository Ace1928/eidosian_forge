from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Experimental(_messages.Message):
    """Experimental service configuration. These configuration options can only
  be used by whitelisted users.

  Fields:
    authorization: Authorization configuration.
  """
    authorization = _messages.MessageField('AuthorizationConfig', 1)