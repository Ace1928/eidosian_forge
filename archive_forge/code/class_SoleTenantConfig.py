from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SoleTenantConfig(_messages.Message):
    """SoleTenantConfig contains the NodeAffinities to specify what shared sole
  tenant node groups should back the node pool.

  Fields:
    nodeAffinities: NodeAffinities used to match to a shared sole tenant node
      group.
  """
    nodeAffinities = _messages.MessageField('NodeAffinity', 1, repeated=True)