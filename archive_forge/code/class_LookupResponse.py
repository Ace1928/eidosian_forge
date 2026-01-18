from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupResponse(_messages.Message):
    """The response for Datastore.Lookup.

  Fields:
    deferred: A list of keys that were not looked up due to resource
      constraints. The order of results in this field is undefined and has no
      relation to the order of the keys in the input.
    found: Entities found as `ResultType.FULL` entities. The order of results
      in this field is undefined and has no relation to the order of the keys
      in the input.
    missing: Entities not found as `ResultType.KEY_ONLY` entities. The order
      of results in this field is undefined and has no relation to the order
      of the keys in the input.
    readTime: The time at which these entities were read or found missing.
    transaction: The identifier of the transaction that was started as part of
      this Lookup request. Set only when ReadOptions.new_transaction was set
      in LookupRequest.read_options.
  """
    deferred = _messages.MessageField('Key', 1, repeated=True)
    found = _messages.MessageField('EntityResult', 2, repeated=True)
    missing = _messages.MessageField('EntityResult', 3, repeated=True)
    readTime = _messages.StringField(4)
    transaction = _messages.BytesField(5)