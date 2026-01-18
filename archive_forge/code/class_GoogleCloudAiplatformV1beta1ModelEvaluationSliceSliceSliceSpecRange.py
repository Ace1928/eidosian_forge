from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpecRange(_messages.Message):
    """A range of values for slice(s). `low` is inclusive, `high` is exclusive.

  Fields:
    high: Exclusive high value for the range.
    low: Inclusive low value for the range.
  """
    high = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    low = _messages.FloatField(2, variant=_messages.Variant.FLOAT)