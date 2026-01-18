from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class AppengineAppUpdateApiClient(base.AppengineApiClientBase):
    """Client used by gcloud to communicate with the App Engine API."""

    def __init__(self, client):
        base.AppengineApiClientBase.__init__(self, client)
        self._registry = resources.REGISTRY.Clone()
        self._registry.RegisterApiByName('appengine', client._VERSION)

    def PatchApplication(self, split_health_checks=None, service_account=None):
        """Updates an application.

    Args:
      split_health_checks: Boolean, whether to enable split health checks by
        default.
      service_account: str, the app-level default service account to update for
        this App Engine app.

    Returns:
      Long running operation.
    """
        update_mask = ''
        if split_health_checks is not None:
            update_mask += 'featureSettings.splitHealthChecks,'
        if service_account is not None:
            update_mask += 'serviceAccount,'
        application_update = self.messages.Application()
        application_update.featureSettings = self.messages.FeatureSettings(splitHealthChecks=split_health_checks)
        application_update.serviceAccount = service_account
        update_request = self.messages.AppengineAppsPatchRequest(name=self._FormatApp(), application=application_update, updateMask=update_mask)
        operation = self.client.apps.Patch(update_request)
        log.debug('Received operation: [{operation}] with mask [{mask}]'.format(operation=operation.name, mask=update_mask))
        return operations_util.WaitForOperation(self.client.apps_operations, operation)