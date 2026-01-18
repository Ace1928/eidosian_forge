from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArimaFittingMetrics(_messages.Message):
    """ARIMA model fitting metrics.

  Fields:
    aic: AIC.
    logLikelihood: Log-likelihood.
    variance: Variance.
  """
    aic = _messages.FloatField(1)
    logLikelihood = _messages.FloatField(2)
    variance = _messages.FloatField(3)