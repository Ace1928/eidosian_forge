from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputationTopology(_messages.Message):
    """All configuration data for a particular Computation.

  Fields:
    computationId: The ID of the computation.
    inputs: The inputs to the computation.
    keyRanges: The key ranges processed by the computation.
    outputs: The outputs from the computation.
    stateFamilies: The state family values.
    systemStageName: The system stage name.
  """
    computationId = _messages.StringField(1)
    inputs = _messages.MessageField('StreamLocation', 2, repeated=True)
    keyRanges = _messages.MessageField('KeyRangeLocation', 3, repeated=True)
    outputs = _messages.MessageField('StreamLocation', 4, repeated=True)
    stateFamilies = _messages.MessageField('StateFamilyConfig', 5, repeated=True)
    systemStageName = _messages.StringField(6)