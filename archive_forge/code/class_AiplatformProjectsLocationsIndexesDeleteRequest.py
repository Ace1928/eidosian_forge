from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexesDeleteRequest object.

  Fields:
    name: Required. The name of the Index resource to be deleted. Format:
      `projects/{project}/locations/{location}/indexes/{index}`
  """
    name = _messages.StringField(1, required=True)