from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PredeployJob(_messages.Message):
    """A predeploy Job.

  Fields:
    actions: Output only. The custom actions that the predeploy Job executes.
  """
    actions = _messages.StringField(1, repeated=True)