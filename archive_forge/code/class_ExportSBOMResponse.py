from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportSBOMResponse(_messages.Message):
    """The response from a call to ExportSBOM.

  Fields:
    discoveryOccurrence: The name of the discovery occurrence in the form
      "projects/{project_id}/occurrences/{OCCURRENCE_ID} It can be used to
      track the progress of the SBOM export.
  """
    discoveryOccurrence = _messages.StringField(1)