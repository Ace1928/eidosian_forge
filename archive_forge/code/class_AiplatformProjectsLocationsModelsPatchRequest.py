from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsPatchRequest object.

  Fields:
    googleCloudAiplatformV1Model: A GoogleCloudAiplatformV1Model resource to
      be passed as the request body.
    name: The resource name of the Model.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see google.protobuf.FieldMask.
  """
    googleCloudAiplatformV1Model = _messages.MessageField('GoogleCloudAiplatformV1Model', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)