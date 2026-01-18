from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsPatchRequest object.

  Fields:
    googleCloudAiplatformV1Dataset: A GoogleCloudAiplatformV1Dataset resource
      to be passed as the request body.
    name: Output only. The resource name of the Dataset.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see google.protobuf.FieldMask. Updatable fields:
      * `display_name` * `description` * `labels`
  """
    googleCloudAiplatformV1Dataset = _messages.MessageField('GoogleCloudAiplatformV1Dataset', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)