from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsRunsPatchRequest
  object.

  Fields:
    googleCloudAiplatformV1TensorboardRun: A
      GoogleCloudAiplatformV1TensorboardRun resource to be passed as the
      request body.
    name: Output only. Name of the TensorboardRun. Format: `projects/{project}
      /locations/{location}/tensorboards/{tensorboard}/experiments/{experiment
      }/runs/{run}`
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the TensorboardRun resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field is overwritten if it's in the mask. If the user does
      not provide a mask then all fields are overwritten if new values are
      specified.
  """
    googleCloudAiplatformV1TensorboardRun = _messages.MessageField('GoogleCloudAiplatformV1TensorboardRun', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)