from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LeaveUntransformed(_messages.Message):
    """Skips the data without modifying it if the requested transformation
  would cause an error. For example, if a `DateShift` transformation were
  applied an an IP address, this mode would leave the IP address unchanged in
  the response.
  """