from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttributesTypeValueValuesEnum(_messages.Enum):
    """Required. Represents the IdP and type of claims that should be
    fetched.

    Values:
      ATTRIBUTES_TYPE_UNSPECIFIED: No AttributesType specified.
      AZURE_AD_GROUPS_MAIL: Used to get the user's group claims from the Azure
        AD identity provider using configuration provided in
        ExtraAttributesOAuth2Client and `mail` property of the
        `microsoft.graph.group` object is used for claim mapping. See
        https://learn.microsoft.com/en-
        us/graph/api/resources/group?view=graph-rest-1.0#properties for more
        details on `microsoft.graph.group` properties. The attributes obtained
        from idntity provider are mapped to `assertion.groups`.
    """
    ATTRIBUTES_TYPE_UNSPECIFIED = 0
    AZURE_AD_GROUPS_MAIL = 1