from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttachedDisk(_messages.Message):
    """An instance-attached disk resource.

  Fields:
    initializeParams: Optional. Specifies the parameters to initialize this
      disk.
  """
    initializeParams = _messages.MessageField('InitializeParams', 1)