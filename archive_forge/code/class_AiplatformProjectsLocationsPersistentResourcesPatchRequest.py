from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPersistentResourcesPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsPersistentResourcesPatchRequest object.

  Fields:
    googleCloudAiplatformV1PersistentResource: A
      GoogleCloudAiplatformV1PersistentResource resource to be passed as the
      request body.
    name: Immutable. Resource name of a PersistentResource.
    updateMask: Required. Specify the fields to be overwritten in the
      PersistentResource by the update method.
  """
    googleCloudAiplatformV1PersistentResource = _messages.MessageField('GoogleCloudAiplatformV1PersistentResource', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)