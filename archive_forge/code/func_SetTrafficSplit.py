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
def SetTrafficSplit(self, service_name, allocations, shard_by='UNSPECIFIED', migrate=False):
    """Sets the traffic split of the given services.

    Args:
      service_name: str, The service name
      allocations: A dict mapping version ID to traffic split.
      shard_by: A ShardByValuesEnum value specifying how to shard the traffic.
      migrate: Whether or not to migrate traffic.
    Returns:
      Long running operation.
    """
    traffic_split = encoding.PyValueToMessage(self.messages.TrafficSplit, {'allocations': allocations, 'shardBy': shard_by})
    update_service_request = self.messages.AppengineAppsServicesPatchRequest(name=self._GetServiceRelativeName(service_name=service_name), service=self.messages.Service(split=traffic_split), migrateTraffic=migrate, updateMask='split')
    message = 'Setting traffic split for service [{service}]'.format(service=service_name)
    operation = self.client.apps_services.Patch(update_service_request)
    return operations_util.WaitForOperation(self.client.apps_operations, operation, message=message)