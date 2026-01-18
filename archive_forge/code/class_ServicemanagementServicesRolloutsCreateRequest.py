from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesRolloutsCreateRequest(_messages.Message):
    """A ServicemanagementServicesRolloutsCreateRequest object.

  Fields:
    force: Optional. This flag will skip safety checks for this rollout. The
      current safety check is whether to skip default quota limit validation.
      Quota limit validation verifies that the default quota limits defined in
      the configs that are effective in this rollout don't decrease by more
      than a specific percentage (10% right now) from the configs that are
      effective in the current rollout. For group-based quota limits, the
      default limit for a quota limit cannot decrease by more than 10%. For
      metric-based quota limits, the assigned quota for each reputation tier
      cannot decrease by more than 10%. Regional quota is assigned per region,
      and the quota for each region cannot decrease by more than 10%. Removing
      a regional quota can cause an effective decrease for that region, if the
      global quota for that tier is lower. For example, if the current rollout
      has a quota limit with values: {STANDARD: 50, STANDARD/us-central1: 100}
      and it is to be changed in the new rollout to: {STANDARD: 50} The net
      effect is the STANDARD tier in us-central1 is decreased by 50%. Adding a
      regional quota can have a similar effect for that region. In order to
      gradually dial down default quota limit, the recommended practice is to
      create multiple rollouts at least 1 hour apart.
    rollout: A Rollout resource to be passed as the request body.
    serviceName: Required. The name of the service. See the
      [overview](https://cloud.google.com/service-management/overview) for
      naming requirements. For example: `example.googleapis.com`.
  """
    force = _messages.BooleanField(1)
    rollout = _messages.MessageField('Rollout', 2)
    serviceName = _messages.StringField(3, required=True)