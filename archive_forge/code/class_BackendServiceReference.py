from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceReference(_messages.Message):
    """A BackendServiceReference object.

  Fields:
    backendService: A string attribute.
  """
    backendService = _messages.StringField(1)