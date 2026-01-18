from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesLunsEvictRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesLunsEvictRequest object.

  Fields:
    evictLunRequest: A EvictLunRequest resource to be passed as the request
      body.
    name: Required. The name of the lun.
  """
    evictLunRequest = _messages.MessageField('EvictLunRequest', 1)
    name = _messages.StringField(2, required=True)