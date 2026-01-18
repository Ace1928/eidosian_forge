from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsProvisioningConfigsSubmitRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsProvisioningConfigsSubmitRequest
  object.

  Fields:
    parent: Required. The parent project and location containing the
      ProvisioningConfig.
    submitProvisioningConfigRequest: A SubmitProvisioningConfigRequest
      resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    submitProvisioningConfigRequest = _messages.MessageField('SubmitProvisioningConfigRequest', 2)