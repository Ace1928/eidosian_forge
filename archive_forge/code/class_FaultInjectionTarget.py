from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FaultInjectionTarget(_messages.Message):
    """Message describing apphub targets passed to Job.

  Fields:
    uri: Uri of the apphub target.
  """
    uri = _messages.StringField(1)