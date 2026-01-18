from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesListRequest(_messages.Message):
    """A BigtableadminProjectsInstancesListRequest object.

  Fields:
    pageToken: DEPRECATED: This field is unused and ignored.
    parent: Required. The unique name of the project for which a list of
      instances is requested. Values are of the form `projects/{project}`.
  """
    pageToken = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)