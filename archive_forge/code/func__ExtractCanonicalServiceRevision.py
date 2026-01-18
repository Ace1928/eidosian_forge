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
def _ExtractCanonicalServiceRevision(workload_labels):
    """Get the canonical service revision of the workload.

  Args:
    workload_labels: A map of workload labels.

  Returns:
    The canonical service revision of the workload.
  """
    if not workload_labels:
        return 'latest'
    rev = workload_labels.get(_ISTIO_CANONICAL_SERVICE_REVISION_LABEL)
    if rev:
        return rev
    rev = workload_labels.get(_KUBERNETES_APP_VERSION_LABEL)
    if rev:
        return rev
    rev = workload_labels.get('version')
    if rev:
        return rev
    return 'latest'