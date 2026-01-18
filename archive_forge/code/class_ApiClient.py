from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.api_gateway import base
from googlecloudsdk.command_lib.api_gateway import common_flags
class ApiClient(base.BaseClient):
    """Client for Api objects on Cloud API Gateway API."""

    def __init__(self, client=None):
        base.BaseClient.__init__(self, client=client, message_base='ApigatewayProjectsLocationsApis', service_name='projects_locations_apis')
        self.DefineGet()
        self.DefineList('apis')
        self.DefineUpdate('apigatewayApi')
        self.DefineDelete()
        self.DefineIamPolicyFunctions()

    def DoesExist(self, api_ref):
        """Checks if an Api object exists.

    Args:
      api_ref: Resource, a resource reference for the api

    Returns:
      Boolean, indicating whether or not exists
    """
        try:
            self.Get(api_ref)
        except apitools_exceptions.HttpNotFoundError:
            return False
        return True

    def Create(self, api_ref, managed_service=None, labels=None, display_name=None):
        """Creates a new Api object.

    Args:
      api_ref: Resource, a resource reference for the api
      managed_service: Optional string, reference name for OP service
      labels: Optional cloud labels
      display_name: Optional display name

    Returns:
      Long running operation response object.
    """
        labels = common_flags.ProcessLabelsFlag(labels, self.messages.ApigatewayApi.LabelsValue)
        api = self.messages.ApigatewayApi(name=api_ref.RelativeName(), managedService=managed_service, labels=labels, displayName=display_name)
        req = self.create_request(apiId=api_ref.Name(), apigatewayApi=api, parent=api_ref.Parent().RelativeName())
        return self.service.Create(req)