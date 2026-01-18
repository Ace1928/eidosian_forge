from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OtherCloudProperties(_messages.Message):
    """Properties specific to this other-cloud (or alternative) provider.

  Enums:
    DataSourceProviderValueValuesEnum: The data source provider of this asset.

  Fields:
    awsDetails: For an asset fetched from AWS.
    connection: The full name of the OtherCloudConnection that is used to
      collect this resource Format:
      `//cloudasset.googleapis.com/organizations//OtherCloudConnections/`
    dataSourceProvider: The data source provider of this asset.
    name: The original name of the resource, such as AWS ARN. It must be able
      to uniquely identify that resource in the data source.
  """

    class DataSourceProviderValueValuesEnum(_messages.Enum):
        """The data source provider of this asset.

    Values:
      PROVIDER_UNSPECIFIED: The unspecified value for data source provider.
      AMAZON_WEB_SERVICES: The value for AWS.
    """
        PROVIDER_UNSPECIFIED = 0
        AMAZON_WEB_SERVICES = 1
    awsDetails = _messages.MessageField('AWSDetails', 1)
    connection = _messages.StringField(2)
    dataSourceProvider = _messages.EnumField('DataSourceProviderValueValuesEnum', 3)
    name = _messages.StringField(4)