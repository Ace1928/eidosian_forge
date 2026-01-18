from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsNodesStartRequest(_messages.Message):
    """A TpuProjectsLocationsNodesStartRequest object.

  Fields:
    name: Required. The resource name.
    startNodeRequest: A StartNodeRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    startNodeRequest = _messages.MessageField('StartNodeRequest', 2)