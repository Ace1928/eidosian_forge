from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesAddSubnetworkRequest(_messages.Message):
    """A ServicenetworkingServicesAddSubnetworkRequest object.

  Fields:
    addSubnetworkRequest: A AddSubnetworkRequest resource to be passed as the
      request body.
    parent: This is a 'tenant' project in the service producer organization.
      services/{service}/{collection-id}/{resource-id} {collection id} is the
      cloud resource collection type representing the tenant project. Only
      'projects' are currently supported. {resource id} is the tenant project
      numeric id: '123456'. {service} the name of the peering service, for
      example 'service.googleapis.com'. This service must be activated in the
      consumer project.
  """
    addSubnetworkRequest = _messages.MessageField('AddSubnetworkRequest', 1)
    parent = _messages.StringField(2, required=True)