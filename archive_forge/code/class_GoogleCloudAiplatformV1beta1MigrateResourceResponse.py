from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigrateResourceResponse(_messages.Message):
    """Describes a successfully migrated resource.

  Fields:
    dataset: Migrated Dataset's resource name.
    migratableResource: Before migration, the identifier in ml.googleapis.com,
      automl.googleapis.com or datalabeling.googleapis.com.
    model: Migrated Model's resource name.
  """
    dataset = _messages.StringField(1)
    migratableResource = _messages.MessageField('GoogleCloudAiplatformV1beta1MigratableResource', 2)
    model = _messages.StringField(3)