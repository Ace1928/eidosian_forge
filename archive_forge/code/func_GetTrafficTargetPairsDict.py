from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
def GetTrafficTargetPairsDict(service_dict):
    """Returns a list of TrafficTargetPairs for a Service as python dictionary.

  Delegates to GetTrafficTargetPairs().

  Args:
    service_dict: python dict-like object representing a Service unmarshalled
      from json
  """
    svc = service.Service(service_dict)
    return GetTrafficTargetPairs(svc.spec_traffic, svc.status_traffic, svc.latest_ready_revision, svc.url)