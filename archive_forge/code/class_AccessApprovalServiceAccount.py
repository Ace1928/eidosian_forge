from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessApprovalServiceAccount(_messages.Message):
    """Access Approval service account related to a
  project/folder/organization.

  Fields:
    accountEmail: Email address of the service account.
    name: The resource name of the Access Approval service account. Format is
      one of: * "projects/{project}/serviceAccount" *
      "folders/{folder}/serviceAccount" *
      "organizations/{organization}/serviceAccount"
  """
    accountEmail = _messages.StringField(1)
    name = _messages.StringField(2)