from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsDatasetsExportDataRequest(_messages.Message):
    """A TranslateProjectsLocationsDatasetsExportDataRequest object.

  Fields:
    dataset: Required. Name of the dataset. In form of `projects/{project-
      number-or-id}/locations/{location-id}/datasets/{dataset-id}`
    exportDataRequest: A ExportDataRequest resource to be passed as the
      request body.
  """
    dataset = _messages.StringField(1, required=True)
    exportDataRequest = _messages.MessageField('ExportDataRequest', 2)