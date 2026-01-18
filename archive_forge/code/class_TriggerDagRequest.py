from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TriggerDagRequest(_messages.Message):
    """Request to trigger a DAG run.

  Messages:
    ConfValue: The key-value pairs get pickled into the conf attribute in the
      DAG run.

  Fields:
    conf: The key-value pairs get pickled into the conf attribute in the DAG
      run.
    dagRunId: The dag_run_id to be assigned to the triggered DAG run.
    executionDate: The execution date of the DAG run.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfValue(_messages.Message):
        """The key-value pairs get pickled into the conf attribute in the DAG
    run.

    Messages:
      AdditionalProperty: An additional property for a ConfValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConfValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    conf = _messages.MessageField('ConfValue', 1)
    dagRunId = _messages.StringField(2)
    executionDate = _messages.StringField(3)