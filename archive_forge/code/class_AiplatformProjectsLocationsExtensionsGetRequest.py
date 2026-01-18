from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsExtensionsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsExtensionsGetRequest object.

  Fields:
    name: Required. The name of the Extension resource. Format:
      `projects/{project}/locations/{location}/extensions/{extension}`
  """
    name = _messages.StringField(1, required=True)