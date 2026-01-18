from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UpdateActiveBreakpointRequest(_messages.Message):
    """Request to update an active breakpoint.

  Fields:
    breakpoint: Required. Updated breakpoint information. The field `id` must
      be set. The agent must echo all Breakpoint specification fields in the
      update.
  """
    breakpoint = _messages.MessageField('Breakpoint', 1)