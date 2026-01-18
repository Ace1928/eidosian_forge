from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplaneV2FeatureState(_messages.Message):
    """An empty state for multi-cluster dataplane-v2 feature. This is required
  since FeatureStateDetails requires a state.
  """