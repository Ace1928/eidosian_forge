from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeRuntimeRequest(_messages.Message):
    """Request for upgrading a Managed Notebook Runtime to the latest version.
  option (google.api.message_visibility).restriction =
  "TRUSTED_TESTER,SPECIAL_TESTER";

  Fields:
    requestId: Idempotent request UUID.
  """
    requestId = _messages.StringField(1)