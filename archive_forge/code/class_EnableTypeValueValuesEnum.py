from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableTypeValueValuesEnum(_messages.Enum):
    """Client and resource project enable type.

    Values:
      ENABLE_TYPE_UNSPECIFIED: Unspecified enable type, which means enabled as
        both client and resource project.
      CLIENT: Enable all clients under the CRM node specified by
        `ConsumerPolicy.name` to use the listed services. A client can be an
        API key, an OAuth client, or a service account.
      RESOURCE: Enable resources in the list services to be created and used
        under the CRM node specified by the `ConsumerPolicy.name`.
      V1_COMPATIBLE: Activation made by Service Usage v1 API. This will be how
        consumers differentiate between policy changes made by v1 and v2
        clients and understand what is actually possible based on those
        different policies.
    """
    ENABLE_TYPE_UNSPECIFIED = 0
    CLIENT = 1
    RESOURCE = 2
    V1_COMPATIBLE = 3