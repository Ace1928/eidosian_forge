from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryPlan(_messages.Message):
    """Contains an ordered list of nodes appearing in the query plan.

  Fields:
    planNodes: The nodes in the query plan. Plan nodes are returned in pre-
      order starting with the plan root. Each PlanNode's `id` corresponds to
      its index in `plan_nodes`.
    queryAdvice: Optional. The advices/recommendations for a query. Currently
      this field will be serving index recommendations for a query.
  """
    planNodes = _messages.MessageField('PlanNode', 1, repeated=True)
    queryAdvice = _messages.MessageField('QueryAdvisorResult', 2)