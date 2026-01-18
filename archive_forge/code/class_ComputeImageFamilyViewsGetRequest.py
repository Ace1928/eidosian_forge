from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeImageFamilyViewsGetRequest(_messages.Message):
    """A ComputeImageFamilyViewsGetRequest object.

  Fields:
    family: Name of the image family to search for.
    project: Project ID for this request.
    zone: The name of the zone for this request.
  """
    family = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)