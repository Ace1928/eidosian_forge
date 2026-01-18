from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import operator
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import exceptions
from googlecloudsdk.api_lib.app import instances_util
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import region_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.third_party.appengine.admin.tools.conversion import convert_yaml
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def _CreateVersion(self, service_name, version_id, service_config, manifest, build, extra_config_settings=None, service_account_email=None):
    """Begins the updates and deployment of new app versions.

    Args:
      service_name: str, The service to deploy.
      version_id: str, The version of the service to deploy.
      service_config: AppInfoExternal, Service info parsed from a service yaml
        file.
      manifest: Dictionary mapping source files to Google Cloud Storage
        locations.
      build: BuildArtifact, a wrapper which contains either the build ID for an
        in-progress parallel build, the name of the container image for a serial
        build, or the options to pass to Appengine for a server-side build.
      extra_config_settings: dict, client config settings to pass to the server
        as beta settings.
      service_account_email: Identity of this deployed version. If not set, the
        Admin API will fall back to use the App Engine default appspot service
        account.

    Returns:
      The Admin API Operation, unfinished.
    """
    version_resource = self._CreateVersionResource(service_config, manifest, version_id, build, extra_config_settings, service_account_email)
    create_request = self.messages.AppengineAppsServicesVersionsCreateRequest(parent=self._GetServiceRelativeName(service_name=service_name), version=version_resource)
    return self.client.apps_services_versions.Create(create_request)