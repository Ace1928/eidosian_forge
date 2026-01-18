from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsExportImageRequest(_messages.Message):
    """A RunProjectsLocationsExportImageRequest object.

  Fields:
    googleCloudRunV2ExportImageRequest: A GoogleCloudRunV2ExportImageRequest
      resource to be passed as the request body.
    name: Required. The name of the resource of which image metadata should be
      exported. Format: `projects/{project_id_or_number}/locations/{location}/
      services/{service}/revisions/{revision}` for Revision `projects/{project
      _id_or_number}/locations/{location}/jobs/{job}/executions/{execution}`
      for Execution
  """
    googleCloudRunV2ExportImageRequest = _messages.MessageField('GoogleCloudRunV2ExportImageRequest', 1)
    name = _messages.StringField(2, required=True)