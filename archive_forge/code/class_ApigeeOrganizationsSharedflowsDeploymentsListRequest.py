from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsDeploymentsListRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsDeploymentsListRequest object.

  Fields:
    parent: Required. Name of the shared flow for which to return deployment
      information in the following format:
      `organizations/{org}/sharedflows/{sharedflow}`
  """
    parent = _messages.StringField(1, required=True)