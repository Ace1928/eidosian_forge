from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceDirectoryFeatureState(_messages.Message):
    """An empty state for service directory feature. This is rqeuired since
  FeatureStateDetails requires a state.
  """