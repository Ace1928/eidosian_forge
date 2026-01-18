from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SearchMigratableResourcesRequest(_messages.Message):
    """Request message for MigrationService.SearchMigratableResources.

  Fields:
    filter: A filter for your search. You can use the following types of
      filters: * Resource type filters. The following strings filter for a
      specific type of MigratableResource: * `ml_engine_model_version:*` *
      `automl_model:*` * `automl_dataset:*` * `data_labeling_dataset:*` *
      "Migrated or not" filters. The following strings filter for resources
      that either have or have not already been migrated: *
      `last_migrate_time:*` filters for migrated resources. * `NOT
      last_migrate_time:*` filters for not yet migrated resources.
    pageSize: The standard page size. The default and maximum value is 100.
    pageToken: The standard page token.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)