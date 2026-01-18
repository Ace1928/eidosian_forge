from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsPreviewsGetRequest(_messages.Message):
    """A ConfigProjectsLocationsPreviewsGetRequest object.

  Fields:
    name: Required. The name of the preview. Format:
      'projects/{project_id}/locations/{location}/previews/{preview}'.
  """
    name = _messages.StringField(1, required=True)