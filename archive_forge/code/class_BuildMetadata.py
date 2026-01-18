from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildMetadata(_messages.Message):
    """A BuildMetadata object.

  Fields:
    finishedOn: A string attribute.
    invocationId: A string attribute.
    startedOn: A string attribute.
  """
    finishedOn = _messages.StringField(1)
    invocationId = _messages.StringField(2)
    startedOn = _messages.StringField(3)