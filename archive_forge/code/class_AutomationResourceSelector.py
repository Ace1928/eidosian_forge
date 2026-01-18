from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomationResourceSelector(_messages.Message):
    """AutomationResourceSelector contains the information to select the
  resources to which an Automation is going to be applied.

  Fields:
    targets: Contains attributes about a target.
  """
    targets = _messages.MessageField('TargetAttribute', 1, repeated=True)