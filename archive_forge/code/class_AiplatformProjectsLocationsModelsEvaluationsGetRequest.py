from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsEvaluationsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsEvaluationsGetRequest object.

  Fields:
    name: Required. The name of the ModelEvaluation resource. Format: `project
      s/{project}/locations/{location}/models/{model}/evaluations/{evaluation}
      `
  """
    name = _messages.StringField(1, required=True)