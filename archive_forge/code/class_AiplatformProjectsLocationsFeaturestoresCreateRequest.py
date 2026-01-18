from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresCreateRequest object.

  Fields:
    featurestoreId: Required. The ID to use for this Featurestore, which will
      become the final component of the Featurestore's resource name. This
      value may be up to 60 characters, and valid characters are `[a-z0-9_]`.
      The first character cannot be a number. The value must be unique within
      the project and location.
    googleCloudAiplatformV1Featurestore: A GoogleCloudAiplatformV1Featurestore
      resource to be passed as the request body.
    parent: Required. The resource name of the Location to create
      Featurestores. Format: `projects/{project}/locations/{location}`
  """
    featurestoreId = _messages.StringField(1)
    googleCloudAiplatformV1Featurestore = _messages.MessageField('GoogleCloudAiplatformV1Featurestore', 2)
    parent = _messages.StringField(3, required=True)