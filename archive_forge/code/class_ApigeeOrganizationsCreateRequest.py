from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsCreateRequest object.

  Fields:
    googleCloudApigeeV1Organization: A GoogleCloudApigeeV1Organization
      resource to be passed as the request body.
    parent: Required. Name of the Google Cloud project in which to associate
      the Apigee organization. Pass the information as a query parameter using
      the following structure in your request: `projects/`
  """
    googleCloudApigeeV1Organization = _messages.MessageField('GoogleCloudApigeeV1Organization', 1)
    parent = _messages.StringField(2)