from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class K8SWorkloadSelector(_messages.Message):
    """For Kubernetes workloads, fleet_member_id is used as workload selector.

  Fields:
    fleetMemberId: Required. Fleet membership ID (only the name part, not the
      full URI). The project and location of the membership are the same as
      the WorkloadRegistration.
  """
    fleetMemberId = _messages.StringField(1)