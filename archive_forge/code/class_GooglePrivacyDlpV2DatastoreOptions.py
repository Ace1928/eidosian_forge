from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DatastoreOptions(_messages.Message):
    """Options defining a data set within Google Cloud Datastore.

  Fields:
    kind: The kind to process.
    partitionId: A partition ID identifies a grouping of entities. The
      grouping is always by project and namespace, however the namespace ID
      may be empty.
  """
    kind = _messages.MessageField('GooglePrivacyDlpV2KindExpression', 1)
    partitionId = _messages.MessageField('GooglePrivacyDlpV2PartitionId', 2)