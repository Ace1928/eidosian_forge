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
class AppEngineService(IapIamResource):
    """IAP IAM App Engine service resource.
  """

    def __init__(self, release_track, project, service_id):
        super(AppEngineService, self).__init__(release_track, project)
        self.service_id = service_id

    def _Name(self):
        return 'App Engine application service'

    def _Parse(self):
        project = _GetProject(self.project)
        return self.registry.Parse(None, params={'project': project.projectNumber, 'iapWebId': _AppEngineAppId(project.projectId), 'serviceId': self.service_id}, collection=IAP_WEB_SERVICES_COLLECTION)