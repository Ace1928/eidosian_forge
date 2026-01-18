from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeerAuthenticationMethod(_messages.Message):
    """[Deprecated] Configuration for the peer authentication method.
  Configuration for the peer authentication method.

  Fields:
    mtls: Set if mTLS is used for peer authentication.
  """
    mtls = _messages.MessageField('MutualTls', 1)