from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersGetRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersGetRequest object.

  Fields:
    name: Required. The unique name of the requested cluster. Values are of
      the form `projects/{project}/instances/{instance}/clusters/{cluster}`.
  """
    name = _messages.StringField(1, required=True)