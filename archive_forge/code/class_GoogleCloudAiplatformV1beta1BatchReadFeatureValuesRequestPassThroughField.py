from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchReadFeatureValuesRequestPassThroughField(_messages.Message):
    """Describe pass-through fields in read_instance source.

  Fields:
    fieldName: Required. The name of the field in the CSV header or the name
      of the column in BigQuery table. The naming restriction is the same as
      Feature.name.
  """
    fieldName = _messages.StringField(1)