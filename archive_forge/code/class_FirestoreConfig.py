from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreConfig(_messages.Message):
    """Message for defining firestore configuration.

  Fields:
    config: Database configuration.
  """
    config = _messages.MessageField('FirestoreDatabaseConfig', 1)