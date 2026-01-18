from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
ValueValueValuesEnum enum type.

        Values:
          ACTIVE: A managed certificate can be provisioned, no issues for this
            domain.
          DOMAIN_STATUS_UNSPECIFIED: <no description>
          FAILED_CAA_CHECKING: Failed to check CAA records for the domain.
          FAILED_CAA_FORBIDDEN: Certificate issuance forbidden by an explicit
            CAA record for the domain.
          FAILED_NOT_VISIBLE: There seems to be problem with the user's DNS or
            load balancer configuration for this domain.
          FAILED_RATE_LIMITED: Reached rate-limit for certificates per top-
            level private domain.
          PROVISIONING: Certificate provisioning for this domain is under way.
            GCP will attempt to provision the first certificate.
        