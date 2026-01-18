from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidatorConfig(_messages.Message):
    """Configuration for validator-related parameters on the beacon client, and
  for any GCP-managed validator client.

  Fields:
    beaconFeeRecipient: An Ethereum address which the beacon client will send
      fee rewards to if no recipient is configured in the validator client.
      See https://lighthouse-book.sigmaprime.io/suggested-fee-recipient.html
      or https://docs.prylabs.network/docs/execution-node/fee-recipient for
      examples of how this is used. Note that while this is often described as
      "suggested", as we run the execution node we can trust the execution
      node, and therefore this is considered enforced.
    managedValidatorClient: Immutable. When true, deploys a GCP-managed
      validator client alongside the beacon client.
    mevRelayUrls: URLs for MEV-relay services to use for block building. When
      set, a GCP-managed MEV-boost service is configured on the beacon client.
  """
    beaconFeeRecipient = _messages.StringField(1)
    managedValidatorClient = _messages.BooleanField(2)
    mevRelayUrls = _messages.StringField(3, repeated=True)