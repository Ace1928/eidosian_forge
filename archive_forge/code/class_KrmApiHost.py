from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KrmApiHost(_messages.Message):
    """A KrmApiHost represents a GKE cluster which is pre-installed with KRM
  resources of services currently supported by the KRM API Hosting API.

  Enums:
    StateValueValuesEnum: Output only. The current state of the internal state
      machine for the KrmApiHost.

  Messages:
    LabelsValue: Labels are used for additional information for a KrmApiHost.

  Fields:
    bundlesConfig: Required. Configuration for the bundles that are enabled on
      the KrmApiHost.
    gkeResourceLink: Output only. KrmApiHost GCP self link used for
      identifying the underlying endpoint (GKE cluster currently).
    labels: Labels are used for additional information for a KrmApiHost.
    managementConfig: Configuration of the cluster management
    name: Output only. The name of this KrmApiHost resource in the format: 'pr
      ojects/{project_id}/locations/{location}/krmApiHosts/{krm_api_host_id}'.
    state: Output only. The current state of the internal state machine for
      the KrmApiHost.
    usePrivateEndpoint: Only allow access to the master's private endpoint IP.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the internal state machine for the
    KrmApiHost.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: KrmApiHost is being created
      RUNNING: KrmApiHost is running
      DELETING: KrmApiHost is being deleted
      SUSPENDED: KrmApiHost is suspended, set on specific wipeout events
      READ_ONLY: KrmApiHost is read only, set on specific abuse & billing
        events
      UPDATING: KrmApiHost is being updated
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        RUNNING = 2
        DELETING = 3
        SUSPENDED = 4
        READ_ONLY = 5
        UPDATING = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels are used for additional information for a KrmApiHost.

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
    bundlesConfig = _messages.MessageField('BundlesConfig', 1)
    gkeResourceLink = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    managementConfig = _messages.MessageField('ManagementConfig', 4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    usePrivateEndpoint = _messages.BooleanField(7)