from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsNodesReimageRequest(_messages.Message):
    """A TpuProjectsLocationsNodesReimageRequest object.

  Fields:
    name: The resource name.
    reimageNodeRequest: A ReimageNodeRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    reimageNodeRequest = _messages.MessageField('ReimageNodeRequest', 2)