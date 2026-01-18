from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesInstancePartitionsDeleteRequest(_messages.Message):
    """A SpannerProjectsInstancesInstancePartitionsDeleteRequest object.

  Fields:
    etag: Optional. If not empty, the API only deletes the instance partition
      when the etag provided matches the current status of the requested
      instance partition. Otherwise, deletes the instance partition without
      checking the current status of the requested instance partition.
    name: Required. The name of the instance partition to be deleted. Values
      are of the form `projects/{project}/instances/{instance}/instancePartiti
      ons/{instance_partition}`
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)