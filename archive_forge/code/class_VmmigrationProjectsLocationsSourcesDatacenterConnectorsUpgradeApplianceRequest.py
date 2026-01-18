from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesDatacenterConnectorsUpgradeApplianceRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesDatacenterConnectorsUpgradeApplianc
  eRequest object.

  Fields:
    datacenterConnector: Required. The DatacenterConnector name.
    upgradeApplianceRequest: A UpgradeApplianceRequest resource to be passed
      as the request body.
  """
    datacenterConnector = _messages.StringField(1, required=True)
    upgradeApplianceRequest = _messages.MessageField('UpgradeApplianceRequest', 2)