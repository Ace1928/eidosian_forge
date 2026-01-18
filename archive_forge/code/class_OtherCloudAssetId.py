from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OtherCloudAssetId(_messages.Message):
    """An identifier of an other-cloud asset. All fields are case sensitive,
  unless explicitly noted.

  Enums:
    DataSourceProviderValueValuesEnum: The data source provider of this asset.

  Fields:
    assetName: The name of this asset in the data source provider. It is the
      original name of the resource. For AWS assets, use
      [ARN](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-
      arns.html)
    assetType: The type of this asset.
    dataCollector: The data collector party collecting the asset.
    dataSourceProvider: The data source provider of this asset.
  """

    class DataSourceProviderValueValuesEnum(_messages.Enum):
        """The data source provider of this asset.

    Values:
      PROVIDER_UNSPECIFIED: The unspecified value for data source provider.
      AMAZON_WEB_SERVICES: The value for AWS.
    """
        PROVIDER_UNSPECIFIED = 0
        AMAZON_WEB_SERVICES = 1
    assetName = _messages.StringField(1)
    assetType = _messages.StringField(2)
    dataCollector = _messages.MessageField('DataCollector', 3)
    dataSourceProvider = _messages.EnumField('DataSourceProviderValueValuesEnum', 4)