from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def GetGkePod(pod_substr=None, kubectl_namespace=None):
    """Returns the name of a running pod in a GKE cluster.

  Retrieves pods in the GKE cluster pointed to by the current kubeconfig
  context. To target a specific cluster, this command should be called within
  the context of a TemporaryKubeconfig context manager.

  If pod_substr is not None, the name of an arbitrary running pod
  whose name contains pod_substr is returned; if no pod's name contains
  pod_substr, an Error is raised. If pod_substr is None, an arbitrary running
  pod is returned.

  Pods with 'Ready: true' condition state are preferred. If there are no such
  pods, any running pod will be returned.

  Args:
    pod_substr: string, a filter to apply to pods. The returned pod name must
      contain pod_substr (if it is not None).
    kubectl_namespace: string or None, namespace to query for gke pods

  Raises:
    Error: if GKE pods cannot be retrieved or desired pod is not found.
  """
    pod_out = io.StringIO()
    args = ['get', 'pods', '--output', 'jsonpath={range .items[*]}{.metadata.name}{"\\t"}{.status.phase}{"\\t"}{.status.conditions[?(.type=="Ready")].status}{"\\n"}']
    try:
        RunKubectlCommand(args, out_func=pod_out.write, err_func=log.err.write, namespace=kubectl_namespace)
    except KubectlError as e:
        raise Error('Error retrieving GKE pods: %s' % e)
    cluster_pods = [GkePodStatus(*pod_status.split('\t')) for pod_status in pod_out.getvalue().split('\n') if pod_status]
    cluster_pods.sort(key=lambda x: x.isReady.lower() != 'true')
    running_pods = [pod_status.name for pod_status in cluster_pods if pod_status.phase.lower() == 'running']
    if not running_pods:
        raise Error('No running GKE pods found. If the environment was recently started, please wait and retry.')
    if pod_substr is None:
        return running_pods[0]
    try:
        return next((pod for pod in running_pods if pod_substr in pod))
    except StopIteration:
        raise Error('Desired GKE pod not found. If the environment was recently started, please wait and retry.')