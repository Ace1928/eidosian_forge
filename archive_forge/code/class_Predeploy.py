from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Predeploy(_messages.Message):
    """Predeploy contains the predeploy job configuration information.

  Fields:
    actions: Optional. A sequence of Skaffold custom actions to invoke during
      execution of the predeploy job.
  """
    actions = _messages.StringField(1, repeated=True)