from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1EntityIdSelector(_messages.Message):
    """Selector for entityId. Getting ids from the given source.

  Fields:
    csvSource: Source of Csv
    entityIdField: Source column that holds entity IDs. If not provided,
      entity IDs are extracted from the column named entity_id.
  """
    csvSource = _messages.MessageField('GoogleCloudAiplatformV1beta1CsvSource', 1)
    entityIdField = _messages.StringField(2)