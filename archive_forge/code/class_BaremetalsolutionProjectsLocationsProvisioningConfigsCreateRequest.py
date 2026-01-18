from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsProvisioningConfigsCreateRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsProvisioningConfigsCreateRequest
  object.

  Fields:
    email: Optional. Email provided to send a confirmation with provisioning
      config to.
    parent: Required. The parent project and location containing the
      ProvisioningConfig.
    provisioningConfig: A ProvisioningConfig resource to be passed as the
      request body.
  """
    email = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)
    provisioningConfig = _messages.MessageField('ProvisioningConfig', 3)