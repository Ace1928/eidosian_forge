from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsDeleteRequest object.

  Fields:
    name: Required. The name of the Model resource to be deleted. Format:
      `projects/{project}/locations/{location}/models/{model}`
  """
    name = _messages.StringField(1, required=True)