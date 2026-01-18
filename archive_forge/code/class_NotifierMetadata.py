from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotifierMetadata(_messages.Message):
    """NotifierMetadata contains the data which can be used to reference or
  describe this notifier.

  Fields:
    name: The human-readable and user-given name for the notifier. For
      example: "repo-merge-email-notifier".
    notifier: The string representing the name and version of notifier to
      deploy. Expected to be of the form of "/:". For example: "gcr.io/my-
      project/notifiers/smtp:1.2.34".
  """
    name = _messages.StringField(1)
    notifier = _messages.StringField(2)