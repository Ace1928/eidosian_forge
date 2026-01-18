from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def GitLabConfigFromArgs(args):
    """Construct the GitLabConfig resource from the command line args.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    A populated GitLabConfig message.
  """
    messages = GetMessagesModule()
    config = messages.GitLabConfig()
    config.username = args.user_name
    secrets = messages.GitLabSecrets()
    secrets.apiAccessTokenVersion = args.api_access_token_secret_version
    secrets.readAccessTokenVersion = args.read_access_token_secret_version
    secrets.webhookSecretVersion = args.webhook_secret_secret_version
    secrets.apiKeyVersion = args.api_key_secret_version
    if not _IsEmptyMessage(secrets):
        config.secrets = secrets
    enterprise_config = messages.GitLabEnterpriseConfig()
    enterprise_config.hostUri = args.host_uri
    service_directory_config = messages.ServiceDirectoryConfig()
    service_directory_config.service = args.service_directory_service
    enterprise_config.serviceDirectoryConfig = service_directory_config
    if args.IsSpecified('ssl_ca_file'):
        enterprise_config.sslCa = args.ssl_ca_file
    if not _IsEmptyMessage(enterprise_config):
        config.enterpriseConfig = enterprise_config
    return config