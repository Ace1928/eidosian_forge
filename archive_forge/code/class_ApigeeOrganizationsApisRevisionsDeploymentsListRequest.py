from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisRevisionsDeploymentsListRequest(_messages.Message):
    """A ApigeeOrganizationsApisRevisionsDeploymentsListRequest object.

  Fields:
    parent: Required. Name of the API proxy revision for which to return
      deployment information in the following format:
      `organizations/{org}/apis/{api}/revisions/{rev}`.
  """
    parent = _messages.StringField(1, required=True)