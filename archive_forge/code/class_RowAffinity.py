from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RowAffinity(_messages.Message):
    """If enabled, the AFE will route the request based on the row key of the
  request, rather than randomly. Instead, each row key will be assigned to a
  cluster, and will stick to that cluster. If clusters are added or removed,
  then this may affect which row keys stick to which clusters. To avoid this,
  users can specify a group cluster.
  """