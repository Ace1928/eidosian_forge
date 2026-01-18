from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TrialParameter(_messages.Message):
    """A message representing a parameter to be tuned.

  Fields:
    parameterId: Output only. The ID of the parameter. The parameter should be
      defined in StudySpec's Parameters.
    value: Output only. The value of the parameter. `number_value` will be set
      if a parameter defined in StudySpec is in type 'INTEGER', 'DOUBLE' or
      'DISCRETE'. `string_value` will be set if a parameter defined in
      StudySpec is in type 'CATEGORICAL'.
  """
    parameterId = _messages.StringField(1)
    value = _messages.MessageField('extra_types.JsonValue', 2)