from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpecMatchingParentIntValueSpec(_messages.Message):
    """Represents the spec to match integer values from parent parameter.

  Fields:
    values: Matches values of the parent parameter with type 'INTEGER'. All
      values must lie in `integer_value_spec` of parent parameter.
  """
    values = _messages.IntegerField(1, repeated=True)