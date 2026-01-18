from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedActionRbmSuggestedActionDial(_messages.Message):
    """Opens the user's default dialer app with the specified phone number but
  does not dial automatically.

  Fields:
    phoneNumber: Required. The phone number to fill in the default dialer app.
      This field should be in [E.164](https://en.wikipedia.org/wiki/E.164)
      format. An example of a correctly formatted phone number: +15556767888.
  """
    phoneNumber = _messages.StringField(1)