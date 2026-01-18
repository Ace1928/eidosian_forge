from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentLevelValueValuesEnum(_messages.Enum):
    """DeploymentLevel of a blueprint signifies where the blueprint will be
    applied. e.g. [HYDRATION, SINGLE_DEPLOYMENT, MULTI_DEPLOYMENT]

    Values:
      DEPLOYMENT_LEVEL_UNSPECIFIED: Default unspecified deployment level.
      HYDRATION: Blueprints at HYDRATION level cannot be used to create a
        Deployment (A user cannot manually initate deployment of these
        blueprints on orchestration or workload cluster). These blueprints
        stay in a user's private catalog and are configured and deployed by
        TNA automation.
      SINGLE_DEPLOYMENT: Blueprints at SINGLE_DEPLOYMENT level can be a)
        Modified in private catalog. b) Used to create a deployment on
        orchestration cluster by the user, once approved.
      MULTI_DEPLOYMENT: Blueprints at MULTI_DEPLOYMENT level can be a)
        Modified in private catalog. b) Used to create a deployment on
        orchestration cluster which will create further hydrated deployments.
      WORKLOAD_CLUSTER_DEPLOYMENT: Blueprints at WORKLOAD_CLUSTER_DEPLOYMENT
        level can be a) Modified in private catalog. b) Used to create a
        deployment on workload cluster by the user, once approved.
    """
    DEPLOYMENT_LEVEL_UNSPECIFIED = 0
    HYDRATION = 1
    SINGLE_DEPLOYMENT = 2
    MULTI_DEPLOYMENT = 3
    WORKLOAD_CLUSTER_DEPLOYMENT = 4