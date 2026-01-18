from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadSelector(_messages.Message):
    """WorkloadSelector specifies the criteria used to determine if a workload
  is in a WorkloadRegistration. Different workload types have their own
  matching criteria.

  Fields:
    k8sWorkloadSelector: Selects K8S workloads.
  """
    k8sWorkloadSelector = _messages.MessageField('K8SWorkloadSelector', 1)