from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceAuthString(_messages.Message):
    """Instance AUTH string details.

  Fields:
    authString: AUTH string set on the instance.
  """
    authString = _messages.StringField(1)