from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverrideStrategyValueValuesEnum(_messages.Enum):
    """Whether the domain creation should override any existing mappings for
    this domain. By default, overrides are rejected.

    Values:
      UNSPECIFIED_DOMAIN_OVERRIDE_STRATEGY: Strategy unspecified. Defaults to
        STRICT.
      STRICT: Overrides not allowed. If a mapping already exists for the
        specified domain, the request will return an ALREADY_EXISTS (409).
      OVERRIDE: Overrides allowed. If a mapping already exists for the
        specified domain, the request will overwrite it. Note that this might
        stop another Google product from serving. For example, if the domain
        is mapped to another App Engine application, that app will no longer
        serve from that domain.
    """
    UNSPECIFIED_DOMAIN_OVERRIDE_STRATEGY = 0
    STRICT = 1
    OVERRIDE = 2