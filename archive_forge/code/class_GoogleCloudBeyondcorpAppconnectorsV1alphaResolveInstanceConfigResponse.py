from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1alphaResolveInstanceConfigResponse(_messages.Message):
    """Response message for BeyondCorp.ResolveInstanceConfig.

  Fields:
    instanceConfig: AppConnectorInstanceConfig.
  """
    instanceConfig = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnectorInstanceConfig', 1)