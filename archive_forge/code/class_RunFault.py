from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunFault(_messages.Message):
    """message to store faults and its durations in experiment

  Fields:
    fault: Fault name to run.
  """
    fault = _messages.StringField(1)