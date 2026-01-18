from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsKuberunsDeleteRequest(_messages.Message):
    """A AnthoseventsKuberunsDeleteRequest object.

  Fields:
    name: The name of the KubeRun resource being deleted.
  """
    name = _messages.StringField(1, required=True)