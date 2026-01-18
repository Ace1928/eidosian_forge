from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrincipalComponentInfo(_messages.Message):
    """Principal component infos, used only for eigen decomposition based
  models, e.g., PCA. Ordered by explained_variance in the descending order.

  Fields:
    cumulativeExplainedVarianceRatio: The explained_variance is pre-ordered in
      the descending order to compute the cumulative explained variance ratio.
    explainedVariance: Explained variance by this principal component, which
      is simply the eigenvalue.
    explainedVarianceRatio: Explained_variance over the total explained
      variance.
    principalComponentId: Id of the principal component.
  """
    cumulativeExplainedVarianceRatio = _messages.FloatField(1)
    explainedVariance = _messages.FloatField(2)
    explainedVarianceRatio = _messages.FloatField(3)
    principalComponentId = _messages.IntegerField(4)