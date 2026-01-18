from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsExtensionsImportRequest(_messages.Message):
    """A AiplatformProjectsLocationsExtensionsImportRequest object.

  Fields:
    googleCloudAiplatformV1beta1Extension: A
      GoogleCloudAiplatformV1beta1Extension resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to import the
      Extension in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1beta1Extension = _messages.MessageField('GoogleCloudAiplatformV1beta1Extension', 1)
    parent = _messages.StringField(2, required=True)