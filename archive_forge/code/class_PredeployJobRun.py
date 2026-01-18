from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PredeployJobRun(_messages.Message):
    """PredeployJobRun contains information specific to a predeploy `JobRun`.

  Enums:
    FailureCauseValueValuesEnum: Output only. The reason the predeploy failed.
      This will always be unspecified while the predeploy is in progress or if
      it succeeded.

  Fields:
    build: Output only. The resource name of the Cloud Build `Build` object
      that is used to execute the custom actions associated with the predeploy
      Job. Format is `projects/{project}/locations/{location}/builds/{build}`.
    failureCause: Output only. The reason the predeploy failed. This will
      always be unspecified while the predeploy is in progress or if it
      succeeded.
    failureMessage: Output only. Additional information about the predeploy
      failure, if available.
  """

    class FailureCauseValueValuesEnum(_messages.Enum):
        """Output only. The reason the predeploy failed. This will always be
    unspecified while the predeploy is in progress or if it succeeded.

    Values:
      FAILURE_CAUSE_UNSPECIFIED: No reason for failure is specified.
      CLOUD_BUILD_UNAVAILABLE: Cloud Build is not available, either because it
        is not enabled or because Cloud Deploy has insufficient permissions.
        See [required permission](https://cloud.google.com/deploy/docs/cloud-
        deploy-service-account#required_permissions).
      EXECUTION_FAILED: The predeploy operation did not complete successfully;
        check Cloud Build logs.
      DEADLINE_EXCEEDED: The predeploy job run did not complete within the
        alloted time.
      CLOUD_BUILD_REQUEST_FAILED: Cloud Build failed to fulfill Cloud Deploy's
        request. See failure_message for additional details.
    """
        FAILURE_CAUSE_UNSPECIFIED = 0
        CLOUD_BUILD_UNAVAILABLE = 1
        EXECUTION_FAILED = 2
        DEADLINE_EXCEEDED = 3
        CLOUD_BUILD_REQUEST_FAILED = 4
    build = _messages.StringField(1)
    failureCause = _messages.EnumField('FailureCauseValueValuesEnum', 2)
    failureMessage = _messages.StringField(3)