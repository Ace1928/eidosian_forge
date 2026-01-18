from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudstoragesourcesReplaceCloudStorageSourceRequest(_messages.Message):
    """A
  AnthoseventsNamespacesCloudstoragesourcesReplaceCloudStorageSourceRequest
  object.

  Fields:
    cloudStorageSource: A CloudStorageSource resource to be passed as the
      request body.
    name: The name of the cloudstoragesource being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    cloudStorageSource = _messages.MessageField('CloudStorageSource', 1)
    name = _messages.StringField(2, required=True)