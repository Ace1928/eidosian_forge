from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Principal(_messages.Message):
    """Users/Service accounts which have access for DNS binding on the intranet
  VPC corresponding to the consumer project.

  Fields:
    serviceAccount: The service account which needs to be granted the
      permission.
    user: The user who needs to be granted permission.
  """
    serviceAccount = _messages.StringField(1)
    user = _messages.StringField(2)