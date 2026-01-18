from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsQueuedResourcesResetRequest(_messages.Message):
    """A TpuProjectsLocationsQueuedResourcesResetRequest object.

  Fields:
    name: Required. The name of the queued resource.
    resetQueuedResourceRequest: A ResetQueuedResourceRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    resetQueuedResourceRequest = _messages.MessageField('ResetQueuedResourceRequest', 2)