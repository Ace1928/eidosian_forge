from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1TreeShapAttribution(_messages.Message):
    """Attributes credit by computing the Shapley value taking advantage of the
  model's tree ensemble structure. Refer to this paper for more details:
  https://arxiv.org/abs/1705.07874 This attribution method is supported for
  XGBoost models.
  """