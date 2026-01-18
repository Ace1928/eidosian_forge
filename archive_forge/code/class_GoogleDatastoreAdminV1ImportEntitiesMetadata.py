from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1ImportEntitiesMetadata(_messages.Message):
    """Metadata for ImportEntities operations.

  Fields:
    common: Metadata common to all Datastore Admin operations.
    entityFilter: Description of which entities are being imported.
    inputUrl: The location of the import metadata file. This will be the same
      value as the google.datastore.admin.v1.ExportEntitiesResponse.output_url
      field.
    progressBytes: An estimate of the number of bytes processed.
    progressEntities: An estimate of the number of entities processed.
  """
    common = _messages.MessageField('GoogleDatastoreAdminV1CommonMetadata', 1)
    entityFilter = _messages.MessageField('GoogleDatastoreAdminV1EntityFilter', 2)
    inputUrl = _messages.StringField(3)
    progressBytes = _messages.MessageField('GoogleDatastoreAdminV1Progress', 4)
    progressEntities = _messages.MessageField('GoogleDatastoreAdminV1Progress', 5)