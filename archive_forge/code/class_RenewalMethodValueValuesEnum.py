from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RenewalMethodValueValuesEnum(_messages.Enum):
    """Output only. The actual renewal method for this `Registration`. When
    `preferred_renewal_method` is set to `AUTOMATIC_RENEWAL`, the actual
    `renewal_method` can be equal to `RENEWAL_DISABLED`-for example, when
    there are problems with the billing account or reported domain abuse. In
    such cases, check the `issues` field on the `Registration`. After the
    problem is resolved, the `renewal_method` is automatically updated to
    `preferred_renewal_method` in a few hours.

    Values:
      RENEWAL_METHOD_UNSPECIFIED: The renewal method is undefined.
      AUTOMATIC_RENEWAL: The domain is automatically renewed each year.
      MANUAL_RENEWAL: Deprecated: For more information, see [Cloud Domains
        feature deprecation](https://cloud.google.com/domains/docs/deprecation
        s/feature-deprecations). This option was never used. Use
        `RENEWAL_DISABLED` instead.
      RENEWAL_DISABLED: The domain won't be renewed and will expire at its
        expiration time.
    """
    RENEWAL_METHOD_UNSPECIFIED = 0
    AUTOMATIC_RENEWAL = 1
    MANUAL_RENEWAL = 2
    RENEWAL_DISABLED = 3