from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
class AppEngineApplication(IapIamResource):
    """IAP IAM App Engine application resource.
  """

    def _Name(self):
        return 'App Engine application'

    def _Parse(self):
        project = _GetProject(self.project)
        return self.registry.Parse(None, params={'project': project.projectNumber, 'iapWebId': _AppEngineAppId(project.projectId)}, collection=IAP_WEB_COLLECTION)

    def _SetAppEngineApplicationIap(self, enabled, oauth2_client_id=None, oauth2_client_secret=None):
        application = _GetApplication(self.project)
        api_client = appengine_api_client.AppengineApiClient.GetApiClient()
        iap_kwargs = _MakeIAPKwargs(False, application.iap, enabled, oauth2_client_id, oauth2_client_secret)
        application_update = api_client.messages.Application(iap=api_client.messages.IdentityAwareProxy(**iap_kwargs))
        application = resources.REGISTRY.Parse(self.project, collection=APPENGINE_APPS_COLLECTION)
        update_request = api_client.messages.AppengineAppsPatchRequest(name=application.RelativeName(), application=application_update, updateMask='iap,')
        operation = api_client.client.apps.Patch(update_request)
        return operations_util.WaitForOperation(api_client.client.apps_operations, operation)

    def Enable(self, oauth2_client_id, oauth2_client_secret):
        """Enable IAP on an App Engine Application."""
        return self._SetAppEngineApplicationIap(True, oauth2_client_id, oauth2_client_secret)

    def Disable(self):
        """Disable IAP on an App Engine Application."""
        return self._SetAppEngineApplicationIap(False)