from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementOperationsGetRequest(_messages.Message):
    """A ServicemanagementOperationsGetRequest object.

  Fields:
    operationsId: Part of `name`. The name of the operation resource.
  """
    operationsId = _messages.StringField(1, required=True)