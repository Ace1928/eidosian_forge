from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationTimestampTransformation(_messages.Message):
    """Training pipeline will perform following transformation functions. *
  Apply the transformation functions for Numerical columns. * Determine the
  year, month, day,and weekday. Treat each value from the * timestamp as a
  Categorical column. * Invalid numerical values (for example, values that
  fall outside of a typical timestamp range, or are extreme values) receive no
  special treatment and are not removed.

  Fields:
    columnName: A string attribute.
    invalidValuesAllowed: If invalid values is allowed, the training pipeline
      will create a boolean feature that indicated whether the value is valid.
      Otherwise, the training pipeline will discard the input row from
      trainining data.
    timeFormat: The format in which that time field is expressed. The
      time_format must either be one of: * `unix-seconds` * `unix-
      milliseconds` * `unix-microseconds` * `unix-nanoseconds` (for
      respectively number of seconds, milliseconds, microseconds and
      nanoseconds since start of the Unix epoch); or be written in `strftime`
      syntax. If time_format is not set, then the default format is RFC 3339
      `date-time` format, where `time-offset` = `"Z"` (e.g.
      1985-04-12T23:20:50.52Z)
  """
    columnName = _messages.StringField(1)
    invalidValuesAllowed = _messages.BooleanField(2)
    timeFormat = _messages.StringField(3)