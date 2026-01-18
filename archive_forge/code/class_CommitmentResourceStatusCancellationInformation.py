from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitmentResourceStatusCancellationInformation(_messages.Message):
    """A CommitmentResourceStatusCancellationInformation object.

  Fields:
    canceledCommitment: [Output Only] An optional amount of CUDs canceled so
      far in the last 365 days.
    canceledCommitmentLastUpdatedTimestamp: [Output Only] An optional last
      update time of canceled_commitment. RFC3339 text format.
    cancellationCap: [Output Only] An optional,the cancellation cap for how
      much commitments can be canceled in a rolling 365 per billing account.
    cancellationFee: [Output Only] An optional, cancellation fee.
    cancellationFeeExpirationTimestamp: [Output Only] An optional,
      cancellation fee expiration time. RFC3339 text format.
  """
    canceledCommitment = _messages.MessageField('Money', 1)
    canceledCommitmentLastUpdatedTimestamp = _messages.StringField(2)
    cancellationCap = _messages.MessageField('Money', 3)
    cancellationFee = _messages.MessageField('Money', 4)
    cancellationFeeExpirationTimestamp = _messages.StringField(5)