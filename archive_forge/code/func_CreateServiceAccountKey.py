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
def CreateServiceAccountKey(service_account_name):
    """Create a service account key.

  Args:
    service_account_name: Name of service acccount.

  Returns:
    The contents of the generated private key file as a string.
  """
    default_credential_path = os.path.join(config.Paths().global_config_dir, _Utf8ToBase64(service_account_name) + '.json')
    credential_file_path = encoding.GetEncodedValue(os.environ, 'LOCAL_CREDENTIAL_PATH', default_credential_path)
    if os.path.exists(credential_file_path):
        return files.ReadFileContents(credential_file_path)
    warning_msg = 'Creating a user-managed service account key for {service_account_name}. This service account key will be the default credential pointed to by GOOGLE_APPLICATION_CREDENTIALS in the local development environment. The user is responsible for the storage,rotation, and deletion of this key. A copy of this key will be stored at {local_key_path}.\nOnly use service accounts from a test project. Do not use service accounts from a production project.'.format(service_account_name=service_account_name, local_key_path=credential_file_path)
    console_io.PromptContinue(message=warning_msg, prompt_string='Continue?', cancel_on_no=True)
    service = apis.GetClientInstance('iam', 'v1')
    message_module = service.MESSAGES_MODULE
    create_key_request = message_module.IamProjectsServiceAccountsKeysCreateRequest(name=service_account_name, createServiceAccountKeyRequest=message_module.CreateServiceAccountKeyRequest(privateKeyType=message_module.CreateServiceAccountKeyRequest.PrivateKeyTypeValueValuesEnum.TYPE_GOOGLE_CREDENTIALS_FILE))
    key = service.projects_serviceAccounts_keys.Create(create_key_request)
    files.WriteFileContents(credential_file_path, key.privateKeyData)
    return six.ensure_text(key.privateKeyData)