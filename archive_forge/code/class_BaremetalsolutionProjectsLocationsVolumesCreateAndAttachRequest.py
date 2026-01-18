from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesCreateAndAttachRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesCreateAndAttachRequest
  object.

  Fields:
    createAndAttachVolumeRequest: A CreateAndAttachVolumeRequest resource to
      be passed as the request body.
    parent: Required. The parent project and location.
  """
    createAndAttachVolumeRequest = _messages.MessageField('CreateAndAttachVolumeRequest', 1)
    parent = _messages.StringField(2, required=True)