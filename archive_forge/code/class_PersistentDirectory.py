from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersistentDirectory(_messages.Message):
    """A directory to persist across workstation sessions.

  Fields:
    gcePd: A PersistentDirectory backed by a Compute Engine persistent disk.
    mountPath: Optional. Location of this directory in the running
      workstation.
  """
    gcePd = _messages.MessageField('GceRegionalPersistentDisk', 1)
    mountPath = _messages.StringField(2)