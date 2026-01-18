from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchImportModelEvaluationSlicesRequest(_messages.Message):
    """Request message for ModelService.BatchImportModelEvaluationSlices

  Fields:
    modelEvaluationSlices: Required. Model evaluation slice resource to be
      imported.
  """
    modelEvaluationSlices = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSlice', 1, repeated=True)