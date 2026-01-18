from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1LoggingConfig(_messages.Message):
    """Parameters that describe the Logging configuration in a cluster.

  Fields:
    componentConfig: The configuration of the logging components;
  """
    componentConfig = _messages.MessageField('GoogleCloudGkemulticloudV1LoggingComponentConfig', 1)