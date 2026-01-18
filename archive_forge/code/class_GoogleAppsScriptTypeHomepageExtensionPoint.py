from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeHomepageExtensionPoint(_messages.Message):
    """Common format for declaring an add-on's homepage view.

  Fields:
    enabled: Optional. If set to `false`, deactivates the homepage view in
      this context. Defaults to `true` if unset. If an add-on's custom
      homepage view is disabled, a generic overview card is provided for users
      instead.
    runFunction: Required. The endpoint to execute when this extension point
      is activated.
  """
    enabled = _messages.BooleanField(1)
    runFunction = _messages.StringField(2)