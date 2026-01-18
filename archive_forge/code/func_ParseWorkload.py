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
def ParseWorkload(workload):
    """Parses the workload value to workload namespace and name.

  Args:
    workload: The workload value with namespace/name format.

  Returns:
    workload namespace and workload name.

  Raises:
    Error: if the workload value is invalid.
  """
    workload_match = re.search(_WORKLOAD_PATTERN, workload)
    if workload_match:
        return (workload_match.group(1), workload_match.group(2))
    raise exceptions.Error('value workload: {} is invalid. Workload value should have the formatnamespace/name.'.format(workload))