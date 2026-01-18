from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformationStatus(_messages.Message):
    """Status of Asset transformation.

  Messages:
    ProgressReportValue: Output only. A struct that was provided by the
      Transformer as progress report.

  Fields:
    invocationId: A UUID of the asset transformation run.
    lastInvocationStatus: Status of the last invocation of the asset
      transformation.
    lastInvocationTime: Output only. Time at which the last invocation of the
      asset transformation occurred.
    progressReport: Output only. A struct that was provided by the Transformer
      as progress report.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProgressReportValue(_messages.Message):
        """Output only. A struct that was provided by the Transformer as progress
    report.

    Messages:
      AdditionalProperty: An additional property for a ProgressReportValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProgressReportValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    invocationId = _messages.StringField(1)
    lastInvocationStatus = _messages.MessageField('Status', 2)
    lastInvocationTime = _messages.StringField(3)
    progressReport = _messages.MessageField('ProgressReportValue', 4)