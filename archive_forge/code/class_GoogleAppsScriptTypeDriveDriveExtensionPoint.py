from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeDriveDriveExtensionPoint(_messages.Message):
    """Common format for declaring a Drive add-on's triggers.

  Fields:
    runFunction: Required. The endpoint to execute when the extension point is
      activated.
  """
    runFunction = _messages.StringField(1)