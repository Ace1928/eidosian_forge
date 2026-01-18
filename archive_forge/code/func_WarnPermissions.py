from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.api_lib.util import http_retry
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import flags
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core.console import console_io
import six.moves.http_client
def WarnPermissions(self, iam_client, messages, permissions, project, organization):
    permissions_helper = util.PermissionsHelper(iam_client, messages, iam_util.GetResourceReference(project, organization), permissions)
    api_disabled_permissions = permissions_helper.GetApiDisabledPermissons()
    iam_util.ApiDisabledPermissionsWarning(api_disabled_permissions)
    testing_permissions = permissions_helper.GetTestingPermissions()
    iam_util.TestingPermissionsWarning(testing_permissions)