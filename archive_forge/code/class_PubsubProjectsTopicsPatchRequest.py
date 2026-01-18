from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsTopicsPatchRequest(_messages.Message):
    """A PubsubProjectsTopicsPatchRequest object.

  Fields:
    name: Required. The name of the topic. It must have the format
      `"projects/{project}/topics/{topic}"`. `{topic}` must start with a
      letter, and contain only letters (`[A-Za-z]`), numbers (`[0-9]`), dashes
      (`-`), underscores (`_`), periods (`.`), tildes (`~`), plus (`+`) or
      percent signs (`%`). It must be between 3 and 255 characters in length,
      and it must not start with `"goog"`.
    updateTopicRequest: A UpdateTopicRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateTopicRequest = _messages.MessageField('UpdateTopicRequest', 2)