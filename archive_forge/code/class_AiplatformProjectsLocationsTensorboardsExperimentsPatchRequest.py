from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsPatchRequest object.

  Fields:
    googleCloudAiplatformV1TensorboardExperiment: A
      GoogleCloudAiplatformV1TensorboardExperiment resource to be passed as
      the request body.
    name: Output only. Name of the TensorboardExperiment. Format: `projects/{p
      roject}/locations/{location}/tensorboards/{tensorboard}/experiments/{exp
      eriment}`
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the TensorboardExperiment resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field is overwritten if it's in the mask. If the
      user does not provide a mask then all fields are overwritten if new
      values are specified.
  """
    googleCloudAiplatformV1TensorboardExperiment = _messages.MessageField('GoogleCloudAiplatformV1TensorboardExperiment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)