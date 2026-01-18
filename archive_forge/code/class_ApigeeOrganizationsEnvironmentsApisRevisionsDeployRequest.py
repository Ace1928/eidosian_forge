from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsApisRevisionsDeployRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsApisRevisionsDeployRequest object.

  Fields:
    name: Required. Name of the API proxy revision deployment in the following
      format:
      `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}`
    override: Flag that specifies whether the new deployment replaces other
      deployed revisions of the API proxy in the environment. Set `override`
      to `true` to replace other deployed revisions. By default, `override` is
      `false` and the deployment is rejected if other revisions of the API
      proxy are deployed in the environment.
    sequencedRollout: Flag that specifies whether to enable sequenced rollout.
      If set to `true`, the routing rules for this deployment and the
      environment changes to add the deployment will be rolled out in a safe
      order. This reduces the risk of downtime that could be caused by
      changing the environment group's routing before the new destination for
      the affected traffic is ready to receive it. This should only be
      necessary if the new deployment will be capturing traffic from another
      environment under a shared environment group or if traffic will be
      rerouted to a different environment due to a base path removal. The
      generateDeployChangeReport API may be used to examine routing changes
      before issuing the deployment request, and its response will indicate if
      a sequenced rollout is recommended for the deployment.
    serviceAccount: Google Cloud IAM service account. The service account
      represents the identity of the deployed proxy, and determines what
      permissions it has. The format must be
      `{ACCOUNT_ID}@{PROJECT}.iam.gserviceaccount.com`.
  """
    name = _messages.StringField(1, required=True)
    override = _messages.BooleanField(2)
    sequencedRollout = _messages.BooleanField(3)
    serviceAccount = _messages.StringField(4)