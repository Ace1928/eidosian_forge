from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersDeleteRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersDeleteRequest object.

  Fields:
    name: Required. The unique name of the cluster to be deleted. Values are
      of the form
      `projects/{project}/instances/{instance}/clusters/{cluster}`.
  """
    name = _messages.StringField(1, required=True)