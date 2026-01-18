from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsExtensionsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsExtensionsDeleteRequest object.

  Fields:
    name: Required. The name of the Extension resource to be deleted. Format:
      `projects/{project}/locations/{location}/extensions/{extension}`
  """
    name = _messages.StringField(1, required=True)