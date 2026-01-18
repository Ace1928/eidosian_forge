from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArimaCoefficients(_messages.Message):
    """Arima coefficients.

  Fields:
    autoRegressiveCoefficients: Auto-regressive coefficients, an array of
      double.
    interceptCoefficient: Intercept coefficient, just a double not an array.
    movingAverageCoefficients: Moving-average coefficients, an array of
      double.
  """
    autoRegressiveCoefficients = _messages.FloatField(1, repeated=True)
    interceptCoefficient = _messages.FloatField(2)
    movingAverageCoefficients = _messages.FloatField(3, repeated=True)