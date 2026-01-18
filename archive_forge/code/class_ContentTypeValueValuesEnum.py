from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContentTypeValueValuesEnum(_messages.Enum):
    """Asset content type. If not specified, no content but the asset name
    and type will be returned.

    Values:
      CONTENT_TYPE_UNSPECIFIED: Unspecified content type.
      RESOURCE: Resource metadata.
      IAM_POLICY: The actual IAM policy set on a resource.
      ORG_POLICY: The organization policy set on an asset.
      ACCESS_POLICY: The Access Context Manager policy set on an asset.
      OS_INVENTORY: The runtime OS Inventory information.
      RELATIONSHIP: The related resources.
    """
    CONTENT_TYPE_UNSPECIFIED = 0
    RESOURCE = 1
    IAM_POLICY = 2
    ORG_POLICY = 3
    ACCESS_POLICY = 4
    OS_INVENTORY = 5
    RELATIONSHIP = 6