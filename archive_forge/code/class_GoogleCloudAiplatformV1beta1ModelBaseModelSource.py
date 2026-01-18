from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelBaseModelSource(_messages.Message):
    """User input field to specify the base model source. Currently it only
  supports specifing the Model Garden models and Genie models.

  Fields:
    genieSource: Information about the base model of Genie models.
    modelGardenSource: Source information of Model Garden models.
  """
    genieSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GenieSource', 1)
    modelGardenSource = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelGardenSource', 2)