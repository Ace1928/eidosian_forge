from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustStore(_messages.Message):
    """Trust store that contains trust anchors and optional intermediate CAs
  used in PKI to build trust chain and verify client's identity.

  Fields:
    intermediateCas: Optional. Set of intermediate CA certificates used for
      building the trust chain to trust anchor.
    trustAnchors: Required. List of Trust Anchors to be used while performing
      validation against a given TrustStore. The incoming end entity's
      certificate must be chained up to one of the trust anchors here.
  """
    intermediateCas = _messages.MessageField('IntermediateCA', 1, repeated=True)
    trustAnchors = _messages.MessageField('TrustAnchor', 2, repeated=True)