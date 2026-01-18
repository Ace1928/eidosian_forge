from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScopeFeatureState(_messages.Message):
    """ScopeFeatureState contains Scope-wide Feature status information.

  Fields:
    helloworld: State for the HelloWorld feature at the scope level
    state: Output only. The "running state" of the Feature in this Scope.
  """
    helloworld = _messages.MessageField('HelloWorldScopeState', 1)
    state = _messages.MessageField('FeatureState', 2)