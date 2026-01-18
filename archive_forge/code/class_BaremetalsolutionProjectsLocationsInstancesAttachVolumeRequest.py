from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesAttachVolumeRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesAttachVolumeRequest object.

  Fields:
    attachVolumeRequest: A AttachVolumeRequest resource to be passed as the
      request body.
    instance: Required. Name of the instance.
  """
    attachVolumeRequest = _messages.MessageField('AttachVolumeRequest', 1)
    instance = _messages.StringField(2, required=True)