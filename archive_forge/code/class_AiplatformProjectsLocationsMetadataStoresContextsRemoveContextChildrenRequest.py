from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsRemoveContextChildrenRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsRemoveContextChildren
  Request object.

  Fields:
    context: Required. The resource name of the parent Context. Format: `proje
      cts/{project}/locations/{location}/metadataStores/{metadatastore}/contex
      ts/{context}`
    googleCloudAiplatformV1RemoveContextChildrenRequest: A
      GoogleCloudAiplatformV1RemoveContextChildrenRequest resource to be
      passed as the request body.
  """
    context = _messages.StringField(1, required=True)
    googleCloudAiplatformV1RemoveContextChildrenRequest = _messages.MessageField('GoogleCloudAiplatformV1RemoveContextChildrenRequest', 2)