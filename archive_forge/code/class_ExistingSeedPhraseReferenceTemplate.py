from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExistingSeedPhraseReferenceTemplate(_messages.Message):
    """Location of the seed material, and derivation path used to generate the
  voting key.

  Fields:
    derivationBase: Optional. The first derivation index to use when deriving
      keys. Must be 0 or greater.
    keyCount: Required. Number of keys (and therefore validators) to derive
      from the seed phrase. Must be between 1 and 1,000.
    seedPhraseSecret: Required. Immutable. Reference into Secret Manager for
      where the seed phrase is stored.
  """
    derivationBase = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    keyCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    seedPhraseSecret = _messages.StringField(3)