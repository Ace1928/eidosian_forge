from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesListRequest(_messages.Message):
    """A ServicemanagementServicesListRequest object.

  Fields:
    category: Include services only in the specified category. Supported
      categories are servicemanagement.googleapis.com/categories/google-
      services or servicemanagement.googleapis.com/categories/play-games.
    consumerProjectId: Include services consumed by the specified project.  If
      project_settings is expanded, then this field controls which project
      project_settings is populated for.
    expand: Fields to expand in any results.  By default, the following fields
      are not fully included in list results: - `operations` -
      `project_settings` - `project_settings.operations` - `quota_usage` (It
      requires `project_settings`)
    pageSize: Requested size of the next page of data.
    pageToken: Token identifying which result to start with; returned by a
      previous list call.
    producerProjectId: Include services produced by the specified project.
  """
    category = _messages.StringField(1)
    consumerProjectId = _messages.StringField(2)
    expand = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    producerProjectId = _messages.StringField(6)