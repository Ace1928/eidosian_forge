from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesGetRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesGetRequest object.

  Fields:
    name: Required. Name of the resource.
  """
    name = _messages.StringField(1, required=True)