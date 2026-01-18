from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSourceProviderValueValuesEnum(_messages.Enum):
    """The data source provider of this asset.

    Values:
      PROVIDER_UNSPECIFIED: The unspecified value for data source provider.
      AMAZON_WEB_SERVICES: The value for AWS.
    """
    PROVIDER_UNSPECIFIED = 0
    AMAZON_WEB_SERVICES = 1