from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacenterConnector(_messages.Message):
    """DatacenterConnector message describes a connector between the Source and
  Google Cloud, which is installed on a vmware datacenter (an OVA vm installed
  by the user) to connect the Datacenter to Google Cloud and support vm
  migration data transfer.

  Enums:
    StateValueValuesEnum: Output only. State of the DatacenterConnector, as
      determined by the health checks.

  Fields:
    applianceInfrastructureVersion: Output only. Appliance OVA version. This
      is the OVA which is manually installed by the user and contains the
      infrastructure for the automatically updatable components on the
      appliance.
    applianceSoftwareVersion: Output only. Appliance last installed update
      bundle version. This is the version of the automatically updatable
      components on the appliance.
    availableVersions: Output only. The available versions for updating this
      appliance.
    bucket: Output only. The communication channel between the datacenter
      connector and Google Cloud.
    createTime: Output only. The time the connector was created (as an API
      call, not when it was actually installed).
    error: Output only. Provides details on the state of the Datacenter
      Connector in case of an error.
    name: Output only. The connector's name.
    registrationId: Immutable. A unique key for this connector. This key is
      internal to the OVA connector and is supplied with its creation during
      the registration process and can not be modified.
    serviceAccount: The service account to use in the connector when
      communicating with the cloud.
    state: Output only. State of the DatacenterConnector, as determined by the
      health checks.
    stateTime: Output only. The time the state was last set.
    updateTime: Output only. The last time the connector was updated with an
      API call.
    upgradeStatus: Output only. The status of the current / last
      upgradeAppliance operation.
    version: The version running in the DatacenterConnector. This is supplied
      by the OVA connector during the registration process and can not be
      modified.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the DatacenterConnector, as determined by the
    health checks.

    Values:
      STATE_UNSPECIFIED: The state is unknown. This is used for API
        compatibility only and is not used by the system.
      PENDING: The state was not sampled by the health checks yet.
      OFFLINE: The source was sampled by health checks and is not available.
      FAILED: The source is available but might not be usable yet due to
        unvalidated credentials or another reason. The credentials referred to
        are the ones to the Source. The error message will contain further
        details.
      ACTIVE: The source exists and its credentials were verified.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        OFFLINE = 2
        FAILED = 3
        ACTIVE = 4
    applianceInfrastructureVersion = _messages.StringField(1)
    applianceSoftwareVersion = _messages.StringField(2)
    availableVersions = _messages.MessageField('AvailableUpdates', 3)
    bucket = _messages.StringField(4)
    createTime = _messages.StringField(5)
    error = _messages.MessageField('Status', 6)
    name = _messages.StringField(7)
    registrationId = _messages.StringField(8)
    serviceAccount = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    stateTime = _messages.StringField(11)
    updateTime = _messages.StringField(12)
    upgradeStatus = _messages.MessageField('UpgradeStatus', 13)
    version = _messages.StringField(14)