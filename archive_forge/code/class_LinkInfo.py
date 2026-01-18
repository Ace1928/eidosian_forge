from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkInfo(_messages.Message):
    """Additional link information.

  Fields:
    owner: Output only. The owner of the link, if it's updated by the system.
  """
    owner = _messages.StringField(1)