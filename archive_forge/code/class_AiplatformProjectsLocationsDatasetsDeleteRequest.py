from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsDeleteRequest object.

  Fields:
    name: Required. The resource name of the Dataset to delete. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`
  """
    name = _messages.StringField(1, required=True)