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
def NamespacesWithLabelSelector(self, label):
    """Get the Connect Agent namespace by label.

    Args:
      label: the label used for namespace selection

    Raises:
      exceptions.Error: if failing to get namespaces.

    Returns:
      The first namespace with the label selector.
    """
    out, err = self._RunKubectl(['get', 'namespaces', '--selector', label, '-o', 'jsonpath={.items}'], None)
    if err:
        raise exceptions.Error('Failed to list namespaces in the cluster: {}'.format(err))
    if out == '[]':
        return []
    out, err = self._RunKubectl(['get', 'namespaces', '--selector', label, '-o', 'jsonpath={.items[0].metadata.name}'], None)
    if err:
        raise exceptions.Error('Failed to list namespaces in the cluster: {}'.format(err))
    return out.strip().split(' ') if out else []