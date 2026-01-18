from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsPreviewsExportRequest(_messages.Message):
    """A ConfigProjectsLocationsPreviewsExportRequest object.

  Fields:
    exportPreviewResultRequest: A ExportPreviewResultRequest resource to be
      passed as the request body.
    parent: Required. The preview whose results should be exported. The
      preview value is in the format:
      'projects/{project_id}/locations/{location}/previews/{preview}'.
  """
    exportPreviewResultRequest = _messages.MessageField('ExportPreviewResultRequest', 1)
    parent = _messages.StringField(2, required=True)