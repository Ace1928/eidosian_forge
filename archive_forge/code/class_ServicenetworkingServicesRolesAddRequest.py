from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesRolesAddRequest(_messages.Message):
    """A ServicenetworkingServicesRolesAddRequest object.

  Fields:
    addRolesRequest: A AddRolesRequest resource to be passed as the request
      body.
    parent: Required. This is in a form services/{service} where {service} is
      the name of the private access management service. For example 'service-
      peering.example.com'.
  """
    addRolesRequest = _messages.MessageField('AddRolesRequest', 1)
    parent = _messages.StringField(2, required=True)