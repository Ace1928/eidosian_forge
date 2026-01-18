from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsExtensionsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsExtensionsPatchRequest object.

  Fields:
    googleCloudAiplatformV1beta1Extension: A
      GoogleCloudAiplatformV1beta1Extension resource to be passed as the
      request body.
    name: Identifier. The resource name of the Extension.
    updateMask: Required. Mask specifying which fields to update. Supported
      fields: * `display_name` * `description` * `tool_use_examples`
  """
    googleCloudAiplatformV1beta1Extension = _messages.MessageField('GoogleCloudAiplatformV1beta1Extension', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)