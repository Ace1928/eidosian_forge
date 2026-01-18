from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsDatasetVersionsRestoreRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsDatasetVersionsRestoreRequest
  object.

  Fields:
    name: Required. The name of the DatasetVersion resource. Format: `projects
      /{project}/locations/{location}/datasets/{dataset}/datasetVersions/{data
      set_version}`
  """
    name = _messages.StringField(1, required=True)