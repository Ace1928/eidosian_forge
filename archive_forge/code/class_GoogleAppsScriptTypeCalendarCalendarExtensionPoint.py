from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeCalendarCalendarExtensionPoint(_messages.Message):
    """Common format for declaring a calendar add-on's triggers.

  Fields:
    runFunction: Required. The endpoint to execute when this extension point
      is activated.
  """
    runFunction = _messages.StringField(1)