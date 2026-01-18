from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventoryZypperPatch(_messages.Message):
    """Details related to a Zypper Patch.

  Fields:
    category: The category of the patch.
    patchName: The name of the patch.
    severity: The severity specified for this patch
    summary: Any summary information provided about this patch.
  """
    category = _messages.StringField(1)
    patchName = _messages.StringField(2)
    severity = _messages.StringField(3)
    summary = _messages.StringField(4)