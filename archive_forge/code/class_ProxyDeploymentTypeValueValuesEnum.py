from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProxyDeploymentTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of the deployment (standard or extensible)
    Deployed proxy revision will be marked as extensible in following 2 cases.
    1. The deployed proxy revision uses extensible policies. 2. If a
    environment supports flowhooks and flow hook is configured.

    Values:
      PROXY_DEPLOYMENT_TYPE_UNSPECIFIED: Default value till public preview.
        After public preview this value should not be returned.
      STANDARD: Deployment will be of type Standard if only Standard proxies
        are used
      EXTENSIBLE: Proxy will be of type Extensible if deployments uses one or
        more Extensible proxies
    """
    PROXY_DEPLOYMENT_TYPE_UNSPECIFIED = 0
    STANDARD = 1
    EXTENSIBLE = 2