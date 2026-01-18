from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesPingsourcesReplacePingSourceRequest(_messages.Message):
    """A AnthoseventsNamespacesPingsourcesReplacePingSourceRequest object.

  Fields:
    name: The name of the pingsource being retrieved. If needed, replace
      {namespace_id} with the project ID.
    pingSource: A PingSource resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    pingSource = _messages.MessageField('PingSource', 2)