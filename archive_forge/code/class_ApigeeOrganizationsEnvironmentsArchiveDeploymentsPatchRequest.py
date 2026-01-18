from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsArchiveDeploymentsPatchRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsArchiveDeploymentsPatchRequest object.

  Fields:
    googleCloudApigeeV1ArchiveDeployment: A
      GoogleCloudApigeeV1ArchiveDeployment resource to be passed as the
      request body.
    name: Name of the Archive Deployment in the following format:
      `organizations/{org}/environments/{env}/archiveDeployments/{id}`.
    updateMask: Required. The list of fields to be updated.
  """
    googleCloudApigeeV1ArchiveDeployment = _messages.MessageField('GoogleCloudApigeeV1ArchiveDeployment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)