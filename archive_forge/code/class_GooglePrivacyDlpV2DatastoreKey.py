from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DatastoreKey(_messages.Message):
    """Record key for a finding in Cloud Datastore.

  Fields:
    entityKey: Datastore entity key.
  """
    entityKey = _messages.MessageField('GooglePrivacyDlpV2Key', 1)