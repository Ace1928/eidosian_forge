from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import os
import re
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from kubernetes import client as kube_client_lib
from kubernetes import config as kube_client_config
from six.moves.urllib.parse import urljoin
def CheckClusterAdminPermissions(self):
    """Check to see if the user can perform all the actions in any namespace.

    Raises:
      KubectlError: if failing to get check for cluster-admin permissions.
      RBACError: if cluster-admin permissions are not found.
    """
    out, err = self._RunKubectl(['auth', 'can-i', '*', '*', '--all-namespaces'], None)
    if err:
        raise KubectlError('Failed to check if the user is a cluster-admin: {}'.format(err))
    if 'yes' not in out:
        raise RBACError('Missing cluster-admin RBAC role: The cluster-admin role-based accesscontrol (RBAC) ClusterRole grants you the cluster permissions necessary to connect your clusters back to Google. \nTo create a ClusterRoleBinding resource in the cluster, run the following command:\n\nkubectl create clusterrolebinding [BINDING_NAME]  --clusterrole cluster-admin --user [USER]')