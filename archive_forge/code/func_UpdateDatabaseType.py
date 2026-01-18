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
def UpdateDatabaseType(self, database_type):
    """Updates an application's database_type.

    Args:
      database_type: New database type to switch to

    Returns:
      Long running operation.
    """
    update_mask = 'databaseType'
    application_update = self.messages.Application()
    application_update.databaseType = database_type
    update_request = self.messages.AppengineAppsPatchRequest(name=self._FormatApp(), application=application_update, updateMask=update_mask)
    operation = self.client.apps.Patch(update_request)
    log.debug('Received operation: [{operation}] with mask [{mask}]'.format(operation=operation.name, mask=update_mask))
    return operations_util.WaitForOperation(self.client.apps_operations, operation)