from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesConfigmapsGetRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesConfigmapsGetRequest object.

  Fields:
    name: Required. The name of the config map being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    name = _messages.StringField(1, required=True)