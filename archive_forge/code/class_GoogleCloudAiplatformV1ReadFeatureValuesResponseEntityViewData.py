from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ReadFeatureValuesResponseEntityViewData(_messages.Message):
    """Container to hold value(s), successive in time, for one Feature from the
  request.

  Fields:
    value: Feature value if a single value is requested.
    values: Feature values list if values, successive in time, are requested.
      If the requested number of values is greater than the number of existing
      Feature values, nonexistent values are omitted instead of being returned
      as empty.
  """
    value = _messages.MessageField('GoogleCloudAiplatformV1FeatureValue', 1)
    values = _messages.MessageField('GoogleCloudAiplatformV1FeatureValueList', 2)