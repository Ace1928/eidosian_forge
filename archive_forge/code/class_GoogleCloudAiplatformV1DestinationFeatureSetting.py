from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DestinationFeatureSetting(_messages.Message):
    """A GoogleCloudAiplatformV1DestinationFeatureSetting object.

  Fields:
    destinationField: Specify the field name in the export destination. If not
      specified, Feature ID is used.
    featureId: Required. The ID of the Feature to apply the setting to.
  """
    destinationField = _messages.StringField(1)
    featureId = _messages.StringField(2)