from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogDescriptor(_messages.Message):
    """A description of a log type. Example in YAML format:      - name:
  library.googleapis.com/activity_history       description: The history of
  borrowing and returning library items.       display_name: Activity
  labels:       - key: /customer_id         description: Identifier of a
  library customer

  Fields:
    description: A human-readable description of this log. This information
      appears in the documentation and can contain details.
    displayName: The human-readable name for this log. This information
      appears on the user interface and should be concise.
    labels: The set of labels that are available to describe a specific log
      entry. Runtime requests that contain labels not specified here are
      considered invalid.
    name: The name of the log. It must be less than 512 characters long and
      can include the following characters: upper- and lower-case alphanumeric
      characters [A-Za-z0-9], and punctuation characters including slash,
      underscore, hyphen, period [/_-.].
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelDescriptor', 3, repeated=True)
    name = _messages.StringField(4)