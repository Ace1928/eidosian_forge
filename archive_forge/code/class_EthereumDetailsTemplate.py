from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EthereumDetailsTemplate(_messages.Message):
    """Blockchain validator configuration unique to Ethereum blockchains.

  Fields:
    gasLimit: Optional. Immutable. Optionally requested (not enforced) maximum
      gas per block. This is sent to the block builder service, however
      whether it is followed depends on the service. This field is only read
      if the field use_block_builder_proposals is set to true. If not
      specified, the validator client will use a default value.
    graffiti: Optional. Input only. Graffiti is a custom string published in
      blocks proposed by the validator. This can only be written, as the
      current value cannot be read back from the validator client API. See
      https://lighthouse-book.sigmaprime.io/graffiti.html for an example of
      how this is used. If not set, the validator client's default is used. If
      no blockchain node is specified, this has no effect as no validator
      client is run.
    suggestedFeeRecipient: Immutable. The Ethereum address to which fee
      rewards should be sent. This can only be set when creating the
      validator. If no blockchain node is specified for the validator, this
      has no effect as no validator client is run. See also
      https://lighthouse-book.sigmaprime.io/suggested-fee-recipient.html for
      more context.
    useBlockBuilderProposals: Optional. Immutable. Enable use of the external
      block building services (MEV).
  """
    gasLimit = _messages.IntegerField(1)
    graffiti = _messages.StringField(2)
    suggestedFeeRecipient = _messages.StringField(3)
    useBlockBuilderProposals = _messages.BooleanField(4)