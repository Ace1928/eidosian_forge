from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SetAddonsRequest(_messages.Message):
    """Request for SetAddons.

  Fields:
    addonsConfig: Required. Add-on configurations.
  """
    addonsConfig = _messages.MessageField('GoogleCloudApigeeV1AddonsConfig', 1)