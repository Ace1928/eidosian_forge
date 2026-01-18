from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV1beta1ServiceIdentity(_messages.Message):
    """Service identity for a service. This is the identity that service
  producer should use to access consumer resources.

  Fields:
    email: The email address of the service account that a service producer
      would use to access consumer resources.
    uniqueId: The unique and stable id of the service account. https://cloud.g
      oogle.com/iam/reference/rest/v1/projects.serviceAccounts#ServiceAccount
  """
    email = _messages.StringField(1)
    uniqueId = _messages.StringField(2)