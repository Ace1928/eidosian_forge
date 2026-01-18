from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotifierConfig(_messages.Message):
    """NotifierConfig is the top-level configuration message.

  Fields:
    apiVersion: The API version of this configuration format.
    kind: The type of notifier to use (e.g. SMTPNotifier).
    metadata: Metadata for referring to/handling/deploying this notifier.
    spec: The actual configuration for this notifier.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('NotifierMetadata', 3)
    spec = _messages.MessageField('NotifierSpec', 4)