from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsEvaluationsSlicesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsEvaluationsSlicesGetRequest object.

  Fields:
    name: Required. The name of the ModelEvaluationSlice resource. Format: `pr
      ojects/{project}/locations/{location}/models/{model}/evaluations/{evalua
      tion}/slices/{slice}`
  """
    name = _messages.StringField(1, required=True)