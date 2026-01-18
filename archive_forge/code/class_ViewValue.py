from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ViewValue(_messages.Message):
    """Information about a logical view.

      Fields:
        privacyPolicy: Specifices the privacy policy for the view.
        useLegacySql: True if view is defined in legacy SQL dialect, false if
          in GoogleSQL.
      """
    privacyPolicy = _messages.MessageField('PrivacyPolicy', 1)
    useLegacySql = _messages.BooleanField(2)