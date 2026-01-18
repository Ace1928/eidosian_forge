from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import base64
import collections
import json
import os
import os.path
import re
import uuid
from apitools.base.py import encoding_helper
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import yaml_parsing as app_engine_yaml_parsing
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import service as k8s_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import secrets
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def CreateDevelopmentServiceAccount(service_account_email):
    """Creates a service account for local development.

  Args:
    service_account_email: Email of the service account.

  Returns:
    The resource name of the service account.
  """
    project_id = _GetServiceAccountProject(service_account_email)
    service_account_name = 'projects/{project}/serviceAccounts/{account}'.format(project=project_id, account=service_account_email)
    exists = _ServiceAccountExists(service_account_name)
    if _IsReservedServiceAccountName(service_account_email):
        if not exists:
            raise ValueError('%s cannot be created because it is a service account name' % service_account_email)
        else:
            return service_account_name
    if not exists:
        account_id = _GetServiceAccountId(service_account_email)
        _CreateAccount('Serverless Local Development Service Account', account_id, project_id)
        permission_msg = 'The project editor role allows the service account to create, delete, and modify most resources in the project.'
        prompt_string = 'Add project editor role to {}?'.format(service_account_email)
        if console_io.PromptContinue(message=permission_msg, prompt_string=prompt_string):
            _AddBinding(project_id, 'serviceAccount:' + service_account_email, 'roles/editor')
    return service_account_name