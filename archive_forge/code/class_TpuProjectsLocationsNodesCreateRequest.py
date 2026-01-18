from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsNodesCreateRequest(_messages.Message):
    """A TpuProjectsLocationsNodesCreateRequest object.

  Fields:
    node: A Node resource to be passed as the request body.
    nodeId: The unqualified resource name.
    parent: Required. The parent resource name.
  """
    node = _messages.MessageField('Node', 1)
    nodeId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)