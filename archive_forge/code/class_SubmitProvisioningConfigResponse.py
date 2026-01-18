from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubmitProvisioningConfigResponse(_messages.Message):
    """Response for SubmitProvisioningConfig.

  Fields:
    provisioningConfig: The submitted provisioning config.
  """
    provisioningConfig = _messages.MessageField('ProvisioningConfig', 1)