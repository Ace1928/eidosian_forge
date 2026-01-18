from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollingStrategy(_messages.Message):
    """RollingStrategy causes a specified number of clusters to be updated
  concurrently until all clusters are updated.

  Fields:
    maxConcurrent: Optional. Maximum number of clusters to update the resource
      bundle on concurrently.
  """
    maxConcurrent = _messages.IntegerField(1, variant=_messages.Variant.INT32)