from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsExportRequest(_messages.Message):
    """A DatastoreProjectsExportRequest object.

  Fields:
    googleDatastoreAdminV1ExportEntitiesRequest: A
      GoogleDatastoreAdminV1ExportEntitiesRequest resource to be passed as the
      request body.
    projectId: Required. Project ID against which to make the request.
  """
    googleDatastoreAdminV1ExportEntitiesRequest = _messages.MessageField('GoogleDatastoreAdminV1ExportEntitiesRequest', 1)
    projectId = _messages.StringField(2, required=True)