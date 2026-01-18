from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExistingSeedPhraseReference(_messages.Message):
    """Location of existing seed material, and derivation path used to generate
  the voting key.

  Fields:
    depositTxData: Output only. Immutable. The deposit transaction data
      corresponding to the derived key.
    derivationIndex: Optional. Immutable. The index to derive the voting key
      at, used as part of a derivation path. The derivation path is built from
      this as "m/12381/3600//0/0" See also
      https://eips.ethereum.org/EIPS/eip-2334#eth2-specific-parameters
    seedPhraseSecret: Required. Immutable. Reference into Secret Manager for
      where the seed phrase is stored.
  """
    depositTxData = _messages.StringField(1)
    derivationIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    seedPhraseSecret = _messages.StringField(3)