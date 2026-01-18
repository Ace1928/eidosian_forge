from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _ExtractCanonicalServiceName(workload_labels, workload_name):
    """Get the canonical service name of the workload.

  Args:
    workload_labels: A map of workload labels.
    workload_name: The name of the workload.

  Returns:
    The canonical service name of the workload.
  """
    if not workload_labels:
        return workload_name
    svc = workload_labels.get(_ISTIO_CANONICAL_SERVICE_NAME_LABEL)
    if svc:
        return svc
    svc = workload_labels.get(_KUBERNETES_APP_NAME_LABEL)
    if svc:
        return svc
    svc = workload_labels.get('app')
    if svc:
        return svc
    return workload_name