from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentTypeValueValuesEnum(_messages.Enum):
    """Optional. Deployment type supported by the environment. The deployment
    type can be set when creating the environment and cannot be changed. When
    you enable archive deployment, you will be **prevented from performing** a
    [subset of actions](/apigee/docs/api-platform/local-
    development/overview#prevented-actions) within the environment, including:
    * Managing the deployment of API proxy or shared flow revisions *
    Creating, updating, or deleting resource files * Creating, updating, or
    deleting target servers

    Values:
      DEPLOYMENT_TYPE_UNSPECIFIED: Deployment type not specified.
      PROXY: Proxy deployment enables you to develop and deploy API proxies
        using Apigee on Google Cloud. This cannot currently be combined with
        the CONFIGURABLE API proxy type.
      ARCHIVE: Archive deployment enables you to develop API proxies locally
        then deploy an archive of your API proxy configuration to an
        environment in Apigee on Google Cloud. You will be prevented from
        performing a [subset of actions](/apigee/docs/api-platform/local-
        development/overview#prevented-actions) within the environment.
    """
    DEPLOYMENT_TYPE_UNSPECIFIED = 0
    PROXY = 1
    ARCHIVE = 2