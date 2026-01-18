from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceGroupReference(_messages.Message):
    """A ResourceGroupReference object.

  Fields:
    group: A URI referencing one of the instance groups or network endpoint
      groups listed in the backend service.
  """
    group = _messages.StringField(1)