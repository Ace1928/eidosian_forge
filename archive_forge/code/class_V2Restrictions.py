from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2Restrictions(_messages.Message):
    """Describes the restrictions on the key.

  Fields:
    androidKeyRestrictions: The Android apps that are allowed to use the key.
    apiTargets: A restriction for a specific service and optionally one or
      more specific methods. Requests are allowed if they match any of these
      restrictions. If no restrictions are specified, all targets are allowed.
    browserKeyRestrictions: The HTTP referrers (websites) that are allowed to
      use the key.
    iosKeyRestrictions: The iOS apps that are allowed to use the key.
    serverKeyRestrictions: The IP addresses of callers that are allowed to use
      the key.
  """
    androidKeyRestrictions = _messages.MessageField('V2AndroidKeyRestrictions', 1)
    apiTargets = _messages.MessageField('V2ApiTarget', 2, repeated=True)
    browserKeyRestrictions = _messages.MessageField('V2BrowserKeyRestrictions', 3)
    iosKeyRestrictions = _messages.MessageField('V2IosKeyRestrictions', 4)
    serverKeyRestrictions = _messages.MessageField('V2ServerKeyRestrictions', 5)