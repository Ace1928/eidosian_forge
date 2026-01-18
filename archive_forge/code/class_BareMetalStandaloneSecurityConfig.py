from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneSecurityConfig(_messages.Message):
    """Specifies the security related settings for the bare metal standalone
  cluster.

  Fields:
    authorization: Configures user access to the standalone cluster.
  """
    authorization = _messages.MessageField('Authorization', 1)