from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatefulPolicyPreservedState(_messages.Message):
    """Configuration of preserved resources.

  Messages:
    DisksValue: Disks created on the instances that will be preserved on
      instance delete, update, etc. This map is keyed with the device names of
      the disks.
    ExternalIPsValue: External network IPs assigned to the instances that will
      be preserved on instance delete, update, etc. This map is keyed with the
      network interface name.
    InternalIPsValue: Internal network IPs assigned to the instances that will
      be preserved on instance delete, update, etc. This map is keyed with the
      network interface name.

  Fields:
    disks: Disks created on the instances that will be preserved on instance
      delete, update, etc. This map is keyed with the device names of the
      disks.
    externalIPs: External network IPs assigned to the instances that will be
      preserved on instance delete, update, etc. This map is keyed with the
      network interface name.
    internalIPs: Internal network IPs assigned to the instances that will be
      preserved on instance delete, update, etc. This map is keyed with the
      network interface name.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DisksValue(_messages.Message):
        """Disks created on the instances that will be preserved on instance
    delete, update, etc. This map is keyed with the device names of the disks.

    Messages:
      AdditionalProperty: An additional property for a DisksValue object.

    Fields:
      additionalProperties: Additional properties of type DisksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DisksValue object.

      Fields:
        key: Name of the additional property.
        value: A StatefulPolicyPreservedStateDiskDevice attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('StatefulPolicyPreservedStateDiskDevice', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExternalIPsValue(_messages.Message):
        """External network IPs assigned to the instances that will be preserved
    on instance delete, update, etc. This map is keyed with the network
    interface name.

    Messages:
      AdditionalProperty: An additional property for a ExternalIPsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExternalIPsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExternalIPsValue object.

      Fields:
        key: Name of the additional property.
        value: A StatefulPolicyPreservedStateNetworkIp attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('StatefulPolicyPreservedStateNetworkIp', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InternalIPsValue(_messages.Message):
        """Internal network IPs assigned to the instances that will be preserved
    on instance delete, update, etc. This map is keyed with the network
    interface name.

    Messages:
      AdditionalProperty: An additional property for a InternalIPsValue
        object.

    Fields:
      additionalProperties: Additional properties of type InternalIPsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InternalIPsValue object.

      Fields:
        key: Name of the additional property.
        value: A StatefulPolicyPreservedStateNetworkIp attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('StatefulPolicyPreservedStateNetworkIp', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    disks = _messages.MessageField('DisksValue', 1)
    externalIPs = _messages.MessageField('ExternalIPsValue', 2)
    internalIPs = _messages.MessageField('InternalIPsValue', 3)