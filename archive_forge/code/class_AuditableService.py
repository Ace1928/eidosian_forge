from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditableService(_messages.Message):
    """Contains information about an auditable service.

  Fields:
    name: Public name of the service. For example, the service name for Cloud
      IAM is 'iam.googleapis.com'.
  """
    name = _messages.StringField(1)