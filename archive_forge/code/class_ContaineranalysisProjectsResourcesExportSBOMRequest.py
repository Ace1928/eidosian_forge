from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsResourcesExportSBOMRequest(_messages.Message):
    """A ContaineranalysisProjectsResourcesExportSBOMRequest object.

  Fields:
    exportSBOMRequest: A ExportSBOMRequest resource to be passed as the
      request body.
    name: Required. The name of the resource in the form of
      `projects/[PROJECT_ID]/resources/[RESOURCE_URL]`.
  """
    exportSBOMRequest = _messages.MessageField('ExportSBOMRequest', 1)
    name = _messages.StringField(2, required=True)