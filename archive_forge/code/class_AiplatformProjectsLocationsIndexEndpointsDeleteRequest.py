from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsDeleteRequest object.

  Fields:
    name: Required. The name of the IndexEndpoint resource to be deleted.
      Format: `projects/{project}/locations/{location}/indexEndpoints/{index_e
      ndpoint}`
  """
    name = _messages.StringField(1, required=True)