from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HcxActivationKey(_messages.Message):
    """HCX activation key. A default key is created during private cloud
  provisioning, but this behavior is subject to change and you should always
  verify active keys. Use VmwareEngine.ListHcxActivationKeys to retrieve
  existing keys and VmwareEngine.CreateHcxActivationKey to create new ones.

  Enums:
    StateValueValuesEnum: Output only. State of HCX activation key.

  Fields:
    activationKey: Output only. HCX activation key.
    createTime: Output only. Creation time of HCX activation key.
    name: Output only. The resource name of this HcxActivationKey. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/privateClouds/my-
      cloud/hcxActivationKeys/my-key`
    state: Output only. State of HCX activation key.
    uid: Output only. System-generated unique identifier for the resource.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of HCX activation key.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      AVAILABLE: State of a newly generated activation key.
      CONSUMED: State of key when it has been used to activate HCX appliance.
      CREATING: State of key when it is being created.
    """
        STATE_UNSPECIFIED = 0
        AVAILABLE = 1
        CONSUMED = 2
        CREATING = 3
    activationKey = _messages.StringField(1)
    createTime = _messages.StringField(2)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    uid = _messages.StringField(5)