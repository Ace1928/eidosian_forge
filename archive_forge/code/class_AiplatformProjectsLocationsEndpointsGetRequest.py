from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsGetRequest object.

  Fields:
    name: Required. The name of the Endpoint resource. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
  """
    name = _messages.StringField(1, required=True)