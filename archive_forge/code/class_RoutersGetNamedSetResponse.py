from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutersGetNamedSetResponse(_messages.Message):
    """A RoutersGetNamedSetResponse object.

  Fields:
    etag: end_interface: MixerGetResponseWithEtagBuilder
    resource: A NamedSet attribute.
  """
    etag = _messages.StringField(1)
    resource = _messages.MessageField('NamedSet', 2)