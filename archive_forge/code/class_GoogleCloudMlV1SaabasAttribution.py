from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1SaabasAttribution(_messages.Message):
    """Attributes credit by running a faster approximation to the TreeShap
  method. Please refer to this link for more details:
  https://blog.datadive.net/interpreting-random-forests/ This attribution
  method is only supported for XGBoost models.
  """