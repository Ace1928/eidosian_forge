from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ImportFeatureValuesRequestFeatureSpec(_messages.Message):
    """Defines the Feature value(s) to import.

  Fields:
    id: Required. ID of the Feature to import values of. This Feature must
      exist in the target EntityType, or the request will fail.
    sourceField: Source column to get the Feature values from. If not set,
      uses the column with the same name as the Feature ID.
  """
    id = _messages.StringField(1)
    sourceField = _messages.StringField(2)