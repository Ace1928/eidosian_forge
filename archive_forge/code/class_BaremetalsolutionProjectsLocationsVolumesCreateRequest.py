from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesCreateRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesCreateRequest object.

  Fields:
    parent: Required. The parent project and location.
    volume: A Volume resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    volume = _messages.MessageField('Volume', 2)