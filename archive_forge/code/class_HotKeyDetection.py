from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HotKeyDetection(_messages.Message):
    """Proto describing a hot key detected on a given WorkItem.

  Fields:
    hotKeyAge: The age of the hot key measured from when it was first
      detected.
    systemName: System-defined name of the step containing this hot key.
      Unique across the workflow.
    userStepName: User-provided name of the step that contains this hot key.
  """
    hotKeyAge = _messages.StringField(1)
    systemName = _messages.StringField(2)
    userStepName = _messages.StringField(3)