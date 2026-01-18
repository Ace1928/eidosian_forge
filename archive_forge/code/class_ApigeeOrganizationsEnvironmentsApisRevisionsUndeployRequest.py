from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsUndeployRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsUndeployRequest object.

  Fields:
    name: Required. Name of the API proxy revision deployment in the following
      format:
      `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}`
    sequencedRollout: Flag that specifies whether to enable sequenced rollout.
      If set to `true`, the environment group routing rules corresponding to
      this deployment will be removed before removing the deployment from the
      runtime. This is likely to be a rare use case; it is only needed when
      the intended effect of undeploying this proxy is to cause the traffic it
      currently handles to be rerouted to some other existing proxy in the
      environment group. The GenerateUndeployChangeReport API may be used to
      examine routing changes before issuing the undeployment request, and its
      response will indicate if a sequenced rollout is recommended for the
      undeployment.
  """
    name = _messages.StringField(1, required=True)
    sequencedRollout = _messages.BooleanField(2)