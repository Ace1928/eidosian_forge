from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersListRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersListRequest object.

  Fields:
    pageToken: DEPRECATED: This field is unused and ignored.
    parent: Required. The unique name of the instance for which a list of
      clusters is requested. Values are of the form
      `projects/{project}/instances/{instance}`. Use `{instance} = '-'` to
      list Clusters for all Instances in a project, e.g.,
      `projects/myproject/instances/-`.
  """
    pageToken = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)