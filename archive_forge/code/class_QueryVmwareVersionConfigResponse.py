from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryVmwareVersionConfigResponse(_messages.Message):
    """Response message for querying VMware user cluster version config.

  Fields:
    versions: List of available versions to install or to upgrade to.
  """
    versions = _messages.MessageField('VmwareVersionInfo', 1, repeated=True)