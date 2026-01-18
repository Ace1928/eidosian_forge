from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class QueryGrantableRolesRequest(_messages.Message):
    """The grantable role query request.

  Fields:
    fullResourceName: Required. The full resource name to query from the list
      of grantable roles.  The name follows the Google Cloud Platform resource
      format. For example, a Cloud Platform project with id `my-project` will
      be named `//cloudresourcemanager.googleapis.com/projects/my-project`.
  """
    fullResourceName = _messages.StringField(1)