from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedDeploymentResourcesTypesValueListEntryValuesEnum(_messages.Enum):
    """SupportedDeploymentResourcesTypesValueListEntryValuesEnum enum type.

    Values:
      DEPLOYMENT_RESOURCES_TYPE_UNSPECIFIED: Should not be used.
      DEDICATED_RESOURCES: Resources that are dedicated to the DeployedModel,
        and that need a higher degree of manual configuration.
      AUTOMATIC_RESOURCES: Resources that to large degree are decided by
        Vertex AI, and require only a modest additional configuration.
      SHARED_RESOURCES: Resources that can be shared by multiple
        DeployedModels. A pre-configured DeploymentResourcePool is required.
    """
    DEPLOYMENT_RESOURCES_TYPE_UNSPECIFIED = 0
    DEDICATED_RESOURCES = 1
    AUTOMATIC_RESOURCES = 2
    SHARED_RESOURCES = 3