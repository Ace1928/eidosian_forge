from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
def GetTrafficTargetPairs(spec_traffic, status_traffic, latest_ready_revision_name, service_url=''):
    """Returns a list of TrafficTargetPairs for a Service.

  Given the spec and status traffic targets wrapped in a TrafficTargets instance
  for a sevice, this function pairs up all spec and status traffic targets that
  reference the same revision (either by name or the latest ready revision) into
  TrafficTargetPairs. This allows the caller to easily see any differences
  between the spec and status traffic.

  Args:
    spec_traffic: A dictionary of name->traffic.TrafficTarget for the spec
      traffic.
    status_traffic: A dictionary of name->traffic.TrafficTarget for the status
      traffic.
    latest_ready_revision_name: The name of the service's latest ready revision.
    service_url: The main URL for the service. Optional.

  Returns:
    A list of TrafficTargetPairs representing the current state of the service's
    traffic assignments. The TrafficTargetPairs are sorted by revision name,
    with targets referencing the latest ready revision at the end.
  """
    spec_dict = dict(spec_traffic)
    status_dict = dict(status_traffic)
    result = []
    for k in set(spec_dict).union(status_dict):
        spec_targets = spec_dict.get(k, [])
        status_targets = status_dict.get(k, [])
        if k == traffic.LATEST_REVISION_KEY:
            revision_name = latest_ready_revision_name
            latest = True
        else:
            revision_name = k
            latest = False
        result.append(TrafficTargetPair(spec_targets, status_targets, revision_name, latest, service_url))
    return sorted(result, key=SortKeyFromTarget)