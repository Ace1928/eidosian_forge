from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneNodePoolConfig(_messages.Message):
    """BareMetalStandaloneNodePoolConfig describes the configuration of all
  nodes within a given bare metal standalone node pool.

  Enums:
    OperatingSystemValueValuesEnum: Specifies the nodes operating system
      (default: LINUX).

  Messages:
    LabelsValue: The labels assigned to nodes of this node pool. An object
      containing a list of key/value pairs. Example: { "name": "wrench",
      "mass": "1.3kg", "count": "3" }.

  Fields:
    kubeletConfig: The modifiable kubelet configurations for the baremetal
      machines.
    labels: The labels assigned to nodes of this node pool. An object
      containing a list of key/value pairs. Example: { "name": "wrench",
      "mass": "1.3kg", "count": "3" }.
    nodeConfigs: Required. The list of machine addresses in the bare metal
      standalone node pool.
    operatingSystem: Specifies the nodes operating system (default: LINUX).
    taints: The initial taints assigned to nodes of this node pool.
  """

    class OperatingSystemValueValuesEnum(_messages.Enum):
        """Specifies the nodes operating system (default: LINUX).

    Values:
      OPERATING_SYSTEM_UNSPECIFIED: No operating system runtime selected.
      LINUX: Linux operating system.
    """
        OPERATING_SYSTEM_UNSPECIFIED = 0
        LINUX = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels assigned to nodes of this node pool. An object containing a
    list of key/value pairs. Example: { "name": "wrench", "mass": "1.3kg",
    "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    kubeletConfig = _messages.MessageField('BareMetalStandaloneKubeletConfig', 1)
    labels = _messages.MessageField('LabelsValue', 2)
    nodeConfigs = _messages.MessageField('BareMetalStandaloneNodeConfig', 3, repeated=True)
    operatingSystem = _messages.EnumField('OperatingSystemValueValuesEnum', 4)
    taints = _messages.MessageField('NodeTaint', 5, repeated=True)