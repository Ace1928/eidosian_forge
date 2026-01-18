from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobDependency(_messages.Message):
    """JobDependency describes the state of other Jobs that the start of this
  Job depends on. All dependent Jobs must have been submitted in the same
  region.

  Messages:
    ItemsValue: Each item maps a Job name to a Type. All items must be
      satisfied for the JobDependency to be satisfied (the AND operation).
      Once a condition for one item becomes true, it won't go back to false
      even the dependent Job state changes again.

  Fields:
    items: Each item maps a Job name to a Type. All items must be satisfied
      for the JobDependency to be satisfied (the AND operation). Once a
      condition for one item becomes true, it won't go back to false even the
      dependent Job state changes again.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ItemsValue(_messages.Message):
        """Each item maps a Job name to a Type. All items must be satisfied for
    the JobDependency to be satisfied (the AND operation). Once a condition
    for one item becomes true, it won't go back to false even the dependent
    Job state changes again.

    Messages:
      AdditionalProperty: An additional property for a ItemsValue object.

    Fields:
      additionalProperties: Additional properties of type ItemsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ItemsValue object.

      Enums:
        ValueValueValuesEnum:

      Fields:
        key: Name of the additional property.
        value: A ValueValueValuesEnum attribute.
      """

            class ValueValueValuesEnum(_messages.Enum):
                """ValueValueValuesEnum enum type.

        Values:
          TYPE_UNSPECIFIED: Unspecified.
          SUCCEEDED: The dependent Job has succeeded.
          FAILED: The dependent Job has failed.
          FINISHED: SUCCEEDED or FAILED.
        """
                TYPE_UNSPECIFIED = 0
                SUCCEEDED = 1
                FAILED = 2
                FINISHED = 3
            key = _messages.StringField(1)
            value = _messages.EnumField('ValueValueValuesEnum', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    items = _messages.MessageField('ItemsValue', 1)