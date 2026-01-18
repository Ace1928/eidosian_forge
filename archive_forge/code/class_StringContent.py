from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StringContent(_messages.Message):
    """String content for a constraint resource.

  Fields:
    yaml: string json = 2;
  """
    yaml = _messages.StringField(1)