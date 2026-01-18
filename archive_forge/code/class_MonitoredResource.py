from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoredResource(_messages.Message):
    """An object representing a resource that can be used for monitoring,
  logging, billing, or other purposes. Examples include virtual machine
  instances, databases, and storage devices such as disks. The type field
  identifies a MonitoredResourceDescriptor object that describes the
  resource's schema. Information in the labels field identifies the actual
  resource and its attributes according to the schema. For example, a
  particular Compute Engine VM instance could be represented by the following
  object, because the MonitoredResourceDescriptor for "gce_instance" has
  labels "project_id", "instance_id" and "zone": { "type": "gce_instance",
  "labels": { "project_id": "my-project", "instance_id": "12345678901234",
  "zone": "us-central1-a" }}

  Messages:
    LabelsValue: Required. Values for all of the labels listed in the
      associated monitored resource descriptor. For example, Compute Engine VM
      instances use the labels "project_id", "instance_id", and "zone".

  Fields:
    labels: Required. Values for all of the labels listed in the associated
      monitored resource descriptor. For example, Compute Engine VM instances
      use the labels "project_id", "instance_id", and "zone".
    type: Required. The monitored resource type. This field must match the
      type field of a MonitoredResourceDescriptor object. For example, the
      type of a Compute Engine VM instance is gce_instance. For a list of
      types, see Monitoring resource types
      (https://cloud.google.com/monitoring/api/resources) and Logging resource
      types (https://cloud.google.com/logging/docs/api/v2/resource-list).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Required. Values for all of the labels listed in the associated
    monitored resource descriptor. For example, Compute Engine VM instances
    use the labels "project_id", "instance_id", and "zone".

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
    labels = _messages.MessageField('LabelsValue', 1)
    type = _messages.StringField(2)