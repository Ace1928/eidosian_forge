from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SlsaBuilder(_messages.Message):
    """A SlsaBuilder object.

  Fields:
    id: A string attribute.
  """
    id = _messages.StringField(1)