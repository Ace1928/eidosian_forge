from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SearchMigratableResourcesResponse(_messages.Message):
    """Response message for MigrationService.SearchMigratableResources.

  Fields:
    migratableResources: All migratable resources that can be migrated to the
      location specified in the request.
    nextPageToken: The standard next-page token. The migratable_resources may
      not fill page_size in SearchMigratableResourcesRequest even when there
      are subsequent pages.
  """
    migratableResources = _messages.MessageField('GoogleCloudAiplatformV1MigratableResource', 1, repeated=True)
    nextPageToken = _messages.StringField(2)