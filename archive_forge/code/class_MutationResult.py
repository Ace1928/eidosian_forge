from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MutationResult(_messages.Message):
    """The result of applying a mutation.

  Fields:
    conflictDetected: Whether a conflict was detected for this mutation.
      Always false when a conflict detection strategy field is not set in the
      mutation.
    createTime: The create time of the entity. This field will not be set
      after a 'delete'.
    key: The automatically allocated key. Set only when the mutation allocated
      a key.
    updateTime: The update time of the entity on the server after processing
      the mutation. If the mutation doesn't change anything on the server,
      then the timestamp will be the update timestamp of the current entity.
      This field will not be set after a 'delete'.
    version: The version of the entity on the server after processing the
      mutation. If the mutation doesn't change anything on the server, then
      the version will be the version of the current entity or, if no entity
      is present, a version that is strictly greater than the version of any
      previous entity and less than the version of any possible future entity.
  """
    conflictDetected = _messages.BooleanField(1)
    createTime = _messages.StringField(2)
    key = _messages.MessageField('Key', 3)
    updateTime = _messages.StringField(4)
    version = _messages.IntegerField(5)