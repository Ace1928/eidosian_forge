from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsQueuedResourcesCreateRequest(_messages.Message):
    """A TpuProjectsLocationsQueuedResourcesCreateRequest object.

  Fields:
    parent: Required. The parent resource name.
    queuedResource: A QueuedResource resource to be passed as the request
      body.
    queuedResourceId: Optional. The unqualified resource name. Should follow
      the `^[A-Za-z0-9_.~+%-]+$` regex format.
    requestId: Optional. Idempotent request UUID.
  """
    parent = _messages.StringField(1, required=True)
    queuedResource = _messages.MessageField('QueuedResource', 2)
    queuedResourceId = _messages.StringField(3)
    requestId = _messages.StringField(4)