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
def _SetBackendServiceIap(self, enabled, oauth2_client_id=None, oauth2_client_secret=None):
    holder = base_classes.ComputeApiHolder(base.ReleaseTrack.GA)
    client = holder.client

    def MakeRequest(method, request):
        return (holder.client.apitools_client.backendServices, method, request)
    backend_service = holder.resources.Parse(self.service_id, params={'project': self.project}, collection=COMPUTE_BACKEND_SERVICES_COLLECTION)
    get_request = client.messages.ComputeBackendServicesGetRequest(project=backend_service.project, backendService=backend_service.Name())
    objects = client.MakeRequests([MakeRequest('Get', get_request)])
    if enabled and objects[0].protocol is not client.messages.BackendService.ProtocolValueValuesEnum.HTTPS:
        log.warning('IAP has been enabled for a backend service that does not use HTTPS. Data sent from the Load Balancer to your VM will not be encrypted.')
    iap_kwargs = _MakeIAPKwargs(True, objects[0].iap, enabled, oauth2_client_id, oauth2_client_secret)
    replacement = encoding.CopyProtoMessage(objects[0])
    replacement.iap = client.messages.BackendServiceIAP(**iap_kwargs)
    update_request = client.messages.ComputeBackendServicesPatchRequest(project=backend_service.project, backendService=backend_service.Name(), backendServiceResource=replacement)
    return client.MakeRequests([MakeRequest('Patch', update_request)])