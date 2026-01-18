from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeProjectsProvisionOrganizationRequest(_messages.Message):
    """A ApigeeProjectsProvisionOrganizationRequest object.

  Fields:
    googleCloudApigeeV1ProvisionOrganizationRequest: A
      GoogleCloudApigeeV1ProvisionOrganizationRequest resource to be passed as
      the request body.
    project: Required. Name of the GCP project with which to associate the
      Apigee organization.
  """
    googleCloudApigeeV1ProvisionOrganizationRequest = _messages.MessageField('GoogleCloudApigeeV1ProvisionOrganizationRequest', 1)
    project = _messages.StringField(2, required=True)