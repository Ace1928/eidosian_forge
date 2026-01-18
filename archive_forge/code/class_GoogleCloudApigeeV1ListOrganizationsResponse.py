from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListOrganizationsResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListOrganizationsResponse object.

  Fields:
    organizations: List of Apigee organizations and associated Google Cloud
      projects.
  """
    organizations = _messages.MessageField('GoogleCloudApigeeV1OrganizationProjectMapping', 1, repeated=True)