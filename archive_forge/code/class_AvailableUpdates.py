from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvailableUpdates(_messages.Message):
    """Holds informatiom about the available versions for upgrade.

  Fields:
    inPlaceUpdate: The latest version for in place update. The current
      appliance can be updated to this version using the API or m4c CLI.
    newDeployableAppliance: The newest deployable version of the appliance.
      The current appliance can't be updated into this version, and the owner
      must manually deploy this OVA to a new appliance.
  """
    inPlaceUpdate = _messages.MessageField('ApplianceVersion', 1)
    newDeployableAppliance = _messages.MessageField('ApplianceVersion', 2)