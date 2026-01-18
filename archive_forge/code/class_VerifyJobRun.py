from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerifyJobRun(_messages.Message):
    """VerifyJobRun contains information specific to a verify `JobRun`.

  Enums:
    FailureCauseValueValuesEnum: Output only. The reason the verify failed.
      This will always be unspecified while the verify is in progress or if it
      succeeded.

  Fields:
    artifactUri: Output only. URI of a directory containing the verify
      artifacts. This contains the Skaffold event log.
    build: Output only. The resource name of the Cloud Build `Build` object
      that is used to verify. Format is
      `projects/{project}/locations/{location}/builds/{build}`.
    eventLogPath: Output only. File path of the Skaffold event log relative to
      the artifact URI.
    failureCause: Output only. The reason the verify failed. This will always
      be unspecified while the verify is in progress or if it succeeded.
    failureMessage: Output only. Additional information about the verify
      failure, if available.
  """

    class FailureCauseValueValuesEnum(_messages.Enum):
        """Output only. The reason the verify failed. This will always be
    unspecified while the verify is in progress or if it succeeded.

    Values:
      FAILURE_CAUSE_UNSPECIFIED: No reason for failure is specified.
      CLOUD_BUILD_UNAVAILABLE: Cloud Build is not available, either because it
        is not enabled or because Cloud Deploy has insufficient permissions.
        See [required permission](https://cloud.google.com/deploy/docs/cloud-
        deploy-service-account#required_permissions).
      EXECUTION_FAILED: The verify operation did not complete successfully;
        check Cloud Build logs.
      DEADLINE_EXCEEDED: The verify job run did not complete within the
        alloted time.
      VERIFICATION_CONFIG_NOT_FOUND: No Skaffold verify configuration was
        found.
      CLOUD_BUILD_REQUEST_FAILED: Cloud Build failed to fulfill Cloud Deploy's
        request. See failure_message for additional details.
    """
        FAILURE_CAUSE_UNSPECIFIED = 0
        CLOUD_BUILD_UNAVAILABLE = 1
        EXECUTION_FAILED = 2
        DEADLINE_EXCEEDED = 3
        VERIFICATION_CONFIG_NOT_FOUND = 4
        CLOUD_BUILD_REQUEST_FAILED = 5
    artifactUri = _messages.StringField(1)
    build = _messages.StringField(2)
    eventLogPath = _messages.StringField(3)
    failureCause = _messages.EnumField('FailureCauseValueValuesEnum', 4)
    failureMessage = _messages.StringField(5)