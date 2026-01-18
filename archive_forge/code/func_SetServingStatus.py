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
def SetServingStatus(self, service_name, version_id, serving_status, block=True):
    """Sets the serving status of the specified version.

    Args:
      service_name: str, The service name
      version_id: str, The version to delete.
      serving_status: The serving status to set.
      block: bool, whether to block on the completion of the operation

    Returns:
      The completed Operation if block is True, or the Operation to wait on
      otherwise.
    """
    patch_request = self.messages.AppengineAppsServicesVersionsPatchRequest(name=self._FormatVersion(service_name=service_name, version_id=version_id), version=self.messages.Version(servingStatus=serving_status), updateMask='servingStatus')
    operation = self.client.apps_services_versions.Patch(patch_request)
    if block:
        return operations_util.WaitForOperation(self.client.apps_operations, operation)
    else:
        return operation