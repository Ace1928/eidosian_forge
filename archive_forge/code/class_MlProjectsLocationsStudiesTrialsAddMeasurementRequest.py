from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsAddMeasurementRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsAddMeasurementRequest object.

  Fields:
    googleCloudMlV1AddTrialMeasurementRequest: A
      GoogleCloudMlV1AddTrialMeasurementRequest resource to be passed as the
      request body.
    name: Required. The trial name.
  """
    googleCloudMlV1AddTrialMeasurementRequest = _messages.MessageField('GoogleCloudMlV1AddTrialMeasurementRequest', 1)
    name = _messages.StringField(2, required=True)