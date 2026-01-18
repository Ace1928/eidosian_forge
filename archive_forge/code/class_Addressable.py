from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Addressable(_messages.Message):
    """Information for connecting over HTTP(s).

  Fields:
    url: A string attribute.
  """
    url = _messages.StringField(1)