from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1ExportEntitiesResponse(_messages.Message):
    """The response for
  google.datastore.admin.v1.DatastoreAdmin.ExportEntities.

  Fields:
    outputUrl: Location of the output metadata file. This can be used to begin
      an import into Cloud Datastore (this project or another project). See
      google.datastore.admin.v1.ImportEntitiesRequest.input_url. Only present
      if the operation completed successfully.
  """
    outputUrl = _messages.StringField(1)