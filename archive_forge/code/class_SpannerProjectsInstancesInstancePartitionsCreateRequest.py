from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesInstancePartitionsCreateRequest(_messages.Message):
    """A SpannerProjectsInstancesInstancePartitionsCreateRequest object.

  Fields:
    createInstancePartitionRequest: A CreateInstancePartitionRequest resource
      to be passed as the request body.
    parent: Required. The name of the instance in which to create the instance
      partition. Values are of the form `projects//instances/`.
  """
    createInstancePartitionRequest = _messages.MessageField('CreateInstancePartitionRequest', 1)
    parent = _messages.StringField(2, required=True)