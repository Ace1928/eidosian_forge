from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigChange(_messages.Message):
    """Output generated from semantically comparing two versions of a service
  configuration. Includes detailed information about a field that have changed
  with applicable advice about potential consequences for the change, such as
  backwards-incompatibility.

  Enums:
    ChangeTypeValueValuesEnum: The type for this change, either ADDED,
      REMOVED, or MODIFIED.

  Fields:
    advices: Collection of advice provided for this change, useful for
      determining the possible impact of this change.
    changeType: The type for this change, either ADDED, REMOVED, or MODIFIED.
    element: Object hierarchy path to the change, with levels separated by a
      '.' character. For repeated fields, an applicable unique identifier
      field is used for the index (usually selector, name, or id). For maps,
      the term 'key' is used. If the field has no unique identifier, the
      numeric index is used. Examples: - visibility.rules[selector=="google.Li
      braryService.ListBooks"].restriction -
      quota.metric_rules[selector=="google"].metric_costs[key=="reads"].value
      - logging.producer_destinations[0]
    newValue: Value of the changed object in the new Service configuration, in
      JSON format. This field will not be populated if ChangeType == REMOVED.
    oldValue: Value of the changed object in the old Service configuration, in
      JSON format. This field will not be populated if ChangeType == ADDED.
  """

    class ChangeTypeValueValuesEnum(_messages.Enum):
        """The type for this change, either ADDED, REMOVED, or MODIFIED.

    Values:
      CHANGE_TYPE_UNSPECIFIED: No value was provided.
      ADDED: The changed object exists in the 'new' service configuration, but
        not in the 'old' service configuration.
      REMOVED: The changed object exists in the 'old' service configuration,
        but not in the 'new' service configuration.
      MODIFIED: The changed object exists in both service configurations, but
        its value is different.
    """
        CHANGE_TYPE_UNSPECIFIED = 0
        ADDED = 1
        REMOVED = 2
        MODIFIED = 3
    advices = _messages.MessageField('Advice', 1, repeated=True)
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 2)
    element = _messages.StringField(3)
    newValue = _messages.StringField(4)
    oldValue = _messages.StringField(5)