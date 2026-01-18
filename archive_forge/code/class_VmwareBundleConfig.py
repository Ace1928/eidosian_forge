from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareBundleConfig(_messages.Message):
    """VmwareBundleConfig represents configuration for the bundle.

  Fields:
    status: Output only. Resource status for the bundle.
    version: The version of the bundle.
  """
    status = _messages.MessageField('ResourceStatus', 1)
    version = _messages.StringField(2)