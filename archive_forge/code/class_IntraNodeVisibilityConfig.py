from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntraNodeVisibilityConfig(_messages.Message):
    """IntraNodeVisibilityConfig contains the desired config of the intra-node
  visibility on this cluster.

  Fields:
    enabled: Enables intra node visibility for this cluster.
  """
    enabled = _messages.BooleanField(1)