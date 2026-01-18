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
def _GetWorkloadLabels(workload_manifest):
    """Get the workload labels from a workload manifest.

  Args:
    workload_manifest: The manifest of the workload.

  Returns:
    The workload labels.

  Raises:
    WorkloadError: If the workload manifest cannot be read.
  """
    if not workload_manifest:
        raise WorkloadError('Cannot verify an empty workload from the cluster')
    try:
        workload_data = yaml.load(workload_manifest)
    except yaml.Error as e:
        raise exceptions.Error('Invalid workload from the cluster {}'.format(workload_data), e)
    workload_labels = _GetNestedKeyFromManifest(workload_data, 'spec', 'metadata', 'labels')
    return workload_labels