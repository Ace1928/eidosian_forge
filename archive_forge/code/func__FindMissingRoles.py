from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def _FindMissingRoles(applied_roles, required_roles):
    """Check which required roles were not covered by given roles.

  Args:
    applied_roles: A set of strings containing the current roles for the
      account.
    required_roles: A set of strings containing the required roles for the
      account.

  Returns:
    A set of missing roles that is not covered.
  """
    if required_roles.issubset(applied_roles):
        return None
    iam_messages = apis.GetMessagesModule('iam', 'v1')
    required_role_permissions = {}
    required_permissions = set()
    applied_permissions = set()
    unsatisfied_roles = set()
    for role in sorted(required_roles):
        request = iam_messages.IamRolesGetRequest(name=role)
        role_permissions = set(apis.GetClientInstance('iam', 'v1').roles.Get(request).includedPermissions)
        required_role_permissions[role] = role_permissions
        required_permissions = required_permissions.union(role_permissions)
    for applied_role in sorted(applied_roles):
        request = iam_messages.IamRolesGetRequest(name=applied_role)
        applied_role_permissions = set(apis.GetClientInstance('iam', 'v1').roles.Get(request).includedPermissions)
        applied_permissions = applied_permissions.union(applied_role_permissions)
    unsatisfied_permissions = required_permissions - applied_permissions
    for role in required_roles:
        if unsatisfied_permissions.intersection(required_role_permissions[role]):
            unsatisfied_roles.add(role)
    return unsatisfied_roles