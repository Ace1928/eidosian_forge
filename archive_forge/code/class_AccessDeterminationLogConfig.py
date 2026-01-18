from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessDeterminationLogConfig(_messages.Message):
    """Configures consent audit log config for FHIR create, read, update, and
  delete (CRUD) operations. Cloud audit log for healthcare API must be
  [enabled](https://cloud.google.com/logging/docs/audit/configure-data-
  access#config-console-enable). The consent-related logs are included as part
  of `protoPayload.metadata`.

  Enums:
    LogLevelValueValuesEnum: Optional. Controls the amount of detail to
      include as part of the audit logs.

  Fields:
    logLevel: Optional. Controls the amount of detail to include as part of
      the audit logs.
  """

    class LogLevelValueValuesEnum(_messages.Enum):
        """Optional. Controls the amount of detail to include as part of the
    audit logs.

    Values:
      LOG_LEVEL_UNSPECIFIED: No log level specified. This value is unused.
      DISABLED: No additional consent-related logging is added to audit logs.
      MINIMUM: The following information is included: * One of the following
        [`consentMode`](https://cloud.google.com/healthcare-api/docs/fhir-
        consent#audit_logs) fields:
        (`off`|`emptyScope`|`enforced`|`btg`|`bypass`). * The accessor's
        request headers * The `log_level` of the AccessDeterminationLogConfig
        * The final consent evaluation (`PERMIT`, `DENY`, or `NO_CONSENT`) * A
        human-readable summary of the evaluation
      VERBOSE: Includes `MINIMUM` and, for each resource owner, returns: * The
        resource owner's name * Most specific part of the `X-Consent-Scope`
        resulting in consensual determination * Timestamp of the applied
        enforcement leading to the decision * Enforcement version at the time
        the applicable consents were applied * The Consent resource name * The
        timestamp of the Consent resource used for enforcement * Policy type
        (`PATIENT` or `ADMIN`) Note that this mode adds some overhead to CRUD
        operations.
    """
        LOG_LEVEL_UNSPECIFIED = 0
        DISABLED = 1
        MINIMUM = 2
        VERBOSE = 3
    logLevel = _messages.EnumField('LogLevelValueValuesEnum', 1)