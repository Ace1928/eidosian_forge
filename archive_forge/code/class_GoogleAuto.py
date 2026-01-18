from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleAuto(_messages.Message):
    """Enables automatic Google account login. If set, the service
  automatically generates a Google test account and adds it to the device,
  before executing the test. Note that test accounts might be reused. Many
  applications show their full set of functionalities when an account is
  present on the device. Logging into the device with these generated accounts
  allows testing more functionalities.
  """