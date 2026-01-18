from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def NodeIdentityOptionsToNodeConfig(options, node_config):
    """Convert node identity options into node config.

  If scopes are specified with the `--scopes` flag, respect them.
  If no scopes are presented, 'gke-default' will be passed here indicating that
  we should use the default set:
  - If no service account is specified, default set is GKE_DEFAULT_SCOPES which
    is handled by ExpandScopeURIs:
    - https://www.googleapis.com/auth/devstorage.read_only,
    - https://www.googleapis.com/auth/logging.write',
    - https://www.googleapis.com/auth/monitoring,
    - https://www.googleapis.com/auth/servicecontrol,
    - https://www.googleapis.com/auth/service.management.readonly,
    - https://www.googleapis.com/auth/trace.append,
  - If a service account is specified, default set is _SERVICE_ACCOUNT_SCOPES:
    - https://www.googleapis.com/auth/cloud-platform
    - https://www.googleapis.com/auth/userinfo.email
  Args:
    options: the CreateCluster or CreateNodePool options.
    node_config: the messages.node_config object to be populated.
  """
    if options.service_account:
        node_config.serviceAccount = options.service_account
        replaced_scopes = []
        for scope in options.scopes:
            if scope == 'gke-default':
                replaced_scopes.extend(_SERVICE_ACCOUNT_SCOPES)
            else:
                replaced_scopes.append(scope)
        options.scopes = replaced_scopes
    options.scopes = ExpandScopeURIs(options.scopes)
    node_config.oauthScopes = sorted(set(options.scopes))