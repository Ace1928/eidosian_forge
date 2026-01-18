from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedirectConditionValueValuesEnum(_messages.Enum):
    """When to redirect sign-ins to the IdP.

    Values:
      REDIRECT_CONDITION_UNSPECIFIED: Default and means "always"
      NEVER: Sign-in flows where the user is prompted for their identity will
        not redirect to the IdP (so the user will most likely be prompted by
        Google for a password), but special flows like IdP-initiated SAML and
        sign-in following automatic redirection to the IdP by domain-specific
        service URLs will accept the IdP's assertion of the user's identity.
    """
    REDIRECT_CONDITION_UNSPECIFIED = 0
    NEVER = 1