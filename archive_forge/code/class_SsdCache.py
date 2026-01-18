from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SsdCache(_messages.Message):
    """SSD Cache to optimize Disk IO capacity usage when using HDD storage.

  Enums:
    StateValueValuesEnum: Output only. The current SsdCache state.

  Messages:
    LabelsValue: Cloud Labels are a flexible and lightweight mechanism for
      organizing cloud resources into groups that reflect a customer's
      organizational needs and deployment strategies. Cloud Labels can be used
      to filter collections of resources. They can be used to control how
      resource metrics are aggregated. And they can be used as arguments to
      policy management rules (e.g. route, firewall, load balancing, etc.). *
      Label keys must be between 1 and 63 characters long and must conform to
      the following regular expression: `a-z{0,62}`. * Label values must be
      between 0 and 63 characters long and must conform to the regular
      expression `[a-z0-9_-]{0,63}`. * No more than 64 labels can be
      associated with a given resource. See https://goo.gl/xmQnxf for more
      information on and examples of labels. If you plan to use labels in your
      own code, please note that additional characters may be allowed in the
      future. And so you are advised to use an internal label representation,
      such as JSON, which doesn't rely upon specific characters being
      disallowed. For example, representing labels as the string: name + "_" +
      value would prove problematic if we were to allow "_" in a future
      release.

  Fields:
    createTime: Output only. The time at which the SsdCache was created.
    displayName: The name of this cache as it appears in UIs.
    labels: Cloud Labels are a flexible and lightweight mechanism for
      organizing cloud resources into groups that reflect a customer's
      organizational needs and deployment strategies. Cloud Labels can be used
      to filter collections of resources. They can be used to control how
      resource metrics are aggregated. And they can be used as arguments to
      policy management rules (e.g. route, firewall, load balancing, etc.). *
      Label keys must be between 1 and 63 characters long and must conform to
      the following regular expression: `a-z{0,62}`. * Label values must be
      between 0 and 63 characters long and must conform to the regular
      expression `[a-z0-9_-]{0,63}`. * No more than 64 labels can be
      associated with a given resource. See https://goo.gl/xmQnxf for more
      information on and examples of labels. If you plan to use labels in your
      own code, please note that additional characters may be allowed in the
      future. And so you are advised to use an internal label representation,
      such as JSON, which doesn't rely upon specific characters being
      disallowed. For example, representing labels as the string: name + "_" +
      value would prove problematic if we were to allow "_" in a future
      release.
    name: A unique identifier for the cache. Values are of the form
      `projects//instanceConfigs//ssdCaches/a-z*[a-z0-9]`. The final segment
      of the name must be between 2 and 64 characters in length. A cache's
      name cannot be changed after the cache is created.
    sizeGib: Optional. Size of SSD cache in GiB.
    state: Output only. The current SsdCache state.
    updateTime: Output only. The time at which the SsdCache was most recently
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current SsdCache state.

    Values:
      STATE_UNSPECIFIED: Not specified.
      CREATING: The SsdCache is still being created. Cache resources may not
        be available yet.
      READY: The SsdCache is fully provisioned and ready to be associated with
        Cloud Spanner instances.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Cloud Labels are a flexible and lightweight mechanism for organizing
    cloud resources into groups that reflect a customer's organizational needs
    and deployment strategies. Cloud Labels can be used to filter collections
    of resources. They can be used to control how resource metrics are
    aggregated. And they can be used as arguments to policy management rules
    (e.g. route, firewall, load balancing, etc.). * Label keys must be between
    1 and 63 characters long and must conform to the following regular
    expression: `a-z{0,62}`. * Label values must be between 0 and 63
    characters long and must conform to the regular expression
    `[a-z0-9_-]{0,63}`. * No more than 64 labels can be associated with a
    given resource. See https://goo.gl/xmQnxf for more information on and
    examples of labels. If you plan to use labels in your own code, please
    note that additional characters may be allowed in the future. And so you
    are advised to use an internal label representation, such as JSON, which
    doesn't rely upon specific characters being disallowed. For example,
    representing labels as the string: name + "_" + value would prove
    problematic if we were to allow "_" in a future release.

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
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    sizeGib = _messages.IntegerField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    updateTime = _messages.StringField(7)