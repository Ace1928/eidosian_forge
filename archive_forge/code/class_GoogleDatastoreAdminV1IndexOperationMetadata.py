from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1IndexOperationMetadata(_messages.Message):
    """Metadata for Index operations.

  Fields:
    common: Metadata common to all Datastore Admin operations.
    indexId: The index resource ID that this operation is acting on.
    progressEntities: An estimate of the number of entities processed.
  """
    common = _messages.MessageField('GoogleDatastoreAdminV1CommonMetadata', 1)
    indexId = _messages.StringField(2)
    progressEntities = _messages.MessageField('GoogleDatastoreAdminV1Progress', 3)