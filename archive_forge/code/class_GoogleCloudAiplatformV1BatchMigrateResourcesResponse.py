from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchMigrateResourcesResponse(_messages.Message):
    """Response message for MigrationService.BatchMigrateResources.

  Fields:
    migrateResourceResponses: Successfully migrated resources.
  """
    migrateResourceResponses = _messages.MessageField('GoogleCloudAiplatformV1MigrateResourceResponse', 1, repeated=True)