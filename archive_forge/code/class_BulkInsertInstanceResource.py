from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BulkInsertInstanceResource(_messages.Message):
    """A transient resource used in compute.instances.bulkInsert and
  compute.regionInstances.bulkInsert . This resource is not persisted
  anywhere, it is used only for processing the requests.

  Messages:
    PerInstancePropertiesValue: Per-instance properties to be set on
      individual instances. Keys of this map specify requested instance names.
      Can be empty if name_pattern is used.

  Fields:
    count: The maximum number of instances to create.
    instanceProperties: The instance properties defining the VM instances to
      be created. Required if sourceInstanceTemplate is not provided.
    locationPolicy: Policy for chosing target zone. For more information, see
      Create VMs in bulk .
    minCount: The minimum number of instances to create. If no min_count is
      specified then count is used as the default value. If min_count
      instances cannot be created, then no instances will be created and
      instances already created will be deleted.
    namePattern: The string pattern used for the names of the VMs. Either
      name_pattern or per_instance_properties must be set. The pattern must
      contain one continuous sequence of placeholder hash characters (#) with
      each character corresponding to one digit of the generated instance
      name. Example: a name_pattern of inst-#### generates instance names such
      as inst-0001 and inst-0002. If existing instances in the same project
      and zone have names that match the name pattern then the generated
      instance numbers start after the biggest existing number. For example,
      if there exists an instance with name inst-0050, then instance names
      generated using the pattern inst-#### begin with inst-0051. The name
      pattern placeholder #...# can contain up to 18 characters.
    perInstanceProperties: Per-instance properties to be set on individual
      instances. Keys of this map specify requested instance names. Can be
      empty if name_pattern is used.
    sourceInstanceTemplate: Specifies the instance template from which to
      create instances. You may combine sourceInstanceTemplate with
      instanceProperties to override specific values from an existing instance
      template. Bulk API follows the semantics of JSON Merge Patch described
      by RFC 7396. It can be a full or partial URL. For example, the following
      are all valid URLs to an instance template: -
      https://www.googleapis.com/compute/v1/projects/project
      /global/instanceTemplates/instanceTemplate -
      projects/project/global/instanceTemplates/instanceTemplate -
      global/instanceTemplates/instanceTemplate This field is optional.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PerInstancePropertiesValue(_messages.Message):
        """Per-instance properties to be set on individual instances. Keys of
    this map specify requested instance names. Can be empty if name_pattern is
    used.

    Messages:
      AdditionalProperty: An additional property for a
        PerInstancePropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type
        PerInstancePropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PerInstancePropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A BulkInsertInstanceResourcePerInstanceProperties attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('BulkInsertInstanceResourcePerInstanceProperties', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    count = _messages.IntegerField(1)
    instanceProperties = _messages.MessageField('InstanceProperties', 2)
    locationPolicy = _messages.MessageField('LocationPolicy', 3)
    minCount = _messages.IntegerField(4)
    namePattern = _messages.StringField(5)
    perInstanceProperties = _messages.MessageField('PerInstancePropertiesValue', 6)
    sourceInstanceTemplate = _messages.StringField(7)