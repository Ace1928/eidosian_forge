from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudstoragesourcesGetRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudstoragesourcesGetRequest object.

  Fields:
    name: The name of the cloudstoragesource being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    name = _messages.StringField(1, required=True)