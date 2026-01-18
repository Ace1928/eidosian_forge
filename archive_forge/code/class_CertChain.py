from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertChain(_messages.Message):
    """A CertChain object.

  Fields:
    certificates: The certificates that form the CA chain, from leaf to root
      order.
  """
    certificates = _messages.StringField(1, repeated=True)