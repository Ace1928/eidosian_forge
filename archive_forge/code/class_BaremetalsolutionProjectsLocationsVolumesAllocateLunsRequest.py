from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesAllocateLunsRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesAllocateLunsRequest object.

  Fields:
    allocateLunsRequest: A AllocateLunsRequest resource to be passed as the
      request body.
    parent: Required. Parent volume.
  """
    allocateLunsRequest = _messages.MessageField('AllocateLunsRequest', 1)
    parent = _messages.StringField(2, required=True)