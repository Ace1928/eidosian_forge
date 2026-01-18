from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadOptions(_messages.Message):
    """The options shared by read requests.

  Enums:
    ReadConsistencyValueValuesEnum: The non-transactional read consistency to
      use.

  Fields:
    newTransaction: Options for beginning a new transaction for this request.
      The new transaction identifier will be returned in the corresponding
      response as either LookupResponse.transaction or
      RunQueryResponse.transaction.
    readConsistency: The non-transactional read consistency to use.
    readTime: Reads entities as they were at the given time. This value is
      only supported for Cloud Firestore in Datastore mode. This must be a
      microsecond precision timestamp within the past one hour, or if Point-
      in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
    transaction: The identifier of the transaction in which to read. A
      transaction identifier is returned by a call to
      Datastore.BeginTransaction.
  """

    class ReadConsistencyValueValuesEnum(_messages.Enum):
        """The non-transactional read consistency to use.

    Values:
      READ_CONSISTENCY_UNSPECIFIED: Unspecified. This value must not be used.
      STRONG: Strong consistency.
      EVENTUAL: Eventual consistency.
    """
        READ_CONSISTENCY_UNSPECIFIED = 0
        STRONG = 1
        EVENTUAL = 2
    newTransaction = _messages.MessageField('TransactionOptions', 1)
    readConsistency = _messages.EnumField('ReadConsistencyValueValuesEnum', 2)
    readTime = _messages.StringField(3)
    transaction = _messages.BytesField(4)