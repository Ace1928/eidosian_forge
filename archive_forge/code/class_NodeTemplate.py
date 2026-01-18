from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeTemplate(_messages.Message):
    """Represent a sole-tenant Node Template resource. You can use a template
  to define properties for nodes in a node group. For more information, read
  Creating node groups and instances.

  Enums:
    CpuOvercommitTypeValueValuesEnum: CPU overcommit.
    StatusValueValuesEnum: [Output Only] The status of the node template. One
      of the following values: CREATING, READY, and DELETING.

  Messages:
    NodeAffinityLabelsValue: Labels to use for node affinity, which will be
      used in instance scheduling.

  Fields:
    accelerators: A AcceleratorConfig attribute.
    cpuOvercommitType: CPU overcommit.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    disks: A LocalDisk attribute.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] The type of the resource. Always compute#nodeTemplate
      for node templates.
    name: The name of the resource, provided by the client when initially
      creating the resource. The resource name must be 1-63 characters long,
      and comply with RFC1035. Specifically, the name must be 1-63 characters
      long and match the regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which
      means the first character must be a lowercase letter, and all following
      characters must be a dash, lowercase letter, or digit, except the last
      character, which cannot be a dash.
    nodeAffinityLabels: Labels to use for node affinity, which will be used in
      instance scheduling.
    nodeType: The node type to use for nodes group that are created from this
      template.
    nodeTypeFlexibility: Do not use. Instead, use the node_type property.
    region: [Output Only] The name of the region where the node template
      resides, such as us-central1.
    selfLink: [Output Only] Server-defined URL for the resource.
    serverBinding: Sets the binding properties for the physical server. Valid
      values include: - *[Default]* RESTART_NODE_ON_ANY_SERVER: Restarts VMs
      on any available physical server - RESTART_NODE_ON_MINIMAL_SERVER:
      Restarts VMs on the same physical server whenever possible See Sole-
      tenant node options for more information.
    status: [Output Only] The status of the node template. One of the
      following values: CREATING, READY, and DELETING.
    statusMessage: [Output Only] An optional, human-readable explanation of
      the status.
  """

    class CpuOvercommitTypeValueValuesEnum(_messages.Enum):
        """CPU overcommit.

    Values:
      CPU_OVERCOMMIT_TYPE_UNSPECIFIED: <no description>
      ENABLED: <no description>
      NONE: <no description>
    """
        CPU_OVERCOMMIT_TYPE_UNSPECIFIED = 0
        ENABLED = 1
        NONE = 2

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the node template. One of the following
    values: CREATING, READY, and DELETING.

    Values:
      CREATING: Resources are being allocated.
      DELETING: The node template is currently being deleted.
      INVALID: Invalid status.
      READY: The node template is ready.
    """
        CREATING = 0
        DELETING = 1
        INVALID = 2
        READY = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NodeAffinityLabelsValue(_messages.Message):
        """Labels to use for node affinity, which will be used in instance
    scheduling.

    Messages:
      AdditionalProperty: An additional property for a NodeAffinityLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        NodeAffinityLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NodeAffinityLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accelerators = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    cpuOvercommitType = _messages.EnumField('CpuOvercommitTypeValueValuesEnum', 2)
    creationTimestamp = _messages.StringField(3)
    description = _messages.StringField(4)
    disks = _messages.MessageField('LocalDisk', 5, repeated=True)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(7, default='compute#nodeTemplate')
    name = _messages.StringField(8)
    nodeAffinityLabels = _messages.MessageField('NodeAffinityLabelsValue', 9)
    nodeType = _messages.StringField(10)
    nodeTypeFlexibility = _messages.MessageField('NodeTemplateNodeTypeFlexibility', 11)
    region = _messages.StringField(12)
    selfLink = _messages.StringField(13)
    serverBinding = _messages.MessageField('ServerBinding', 14)
    status = _messages.EnumField('StatusValueValuesEnum', 15)
    statusMessage = _messages.StringField(16)