from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeployJobRun(_messages.Message):
    """DeployJobRun contains information specific to a deploy `JobRun`.

  Enums:
    FailureCauseValueValuesEnum: Output only. The reason the deploy failed.
      This will always be unspecified while the deploy is in progress or if it
      succeeded.

  Fields:
    artifact: Output only. The artifact of a deploy job run, if available.
    build: Output only. The resource name of the Cloud Build `Build` object
      that is used to deploy. Format is
      `projects/{project}/locations/{location}/builds/{build}`.
    failureCause: Output only. The reason the deploy failed. This will always
      be unspecified while the deploy is in progress or if it succeeded.
    failureMessage: Output only. Additional information about the deploy
      failure, if available.
    metadata: Output only. Metadata containing information about the deploy
      job run.
  """

    class FailureCauseValueValuesEnum(_messages.Enum):
        """Output only. The reason the deploy failed. This will always be
    unspecified while the deploy is in progress or if it succeeded.

    Values:
      FAILURE_CAUSE_UNSPECIFIED: No reason for failure is specified.
      CLOUD_BUILD_UNAVAILABLE: Cloud Build is not available, either because it
        is not enabled or because Cloud Deploy has insufficient permissions.
        See [Required permission](https://cloud.google.com/deploy/docs/cloud-
        deploy-service-account#required_permissions).
      EXECUTION_FAILED: The deploy operation did not complete successfully;
        check Cloud Build logs.
      DEADLINE_EXCEEDED: The deploy job run did not complete within the
        alloted time.
      MISSING_RESOURCES_FOR_CANARY: There were missing resources in the
        runtime environment required for a canary deployment. Check the Cloud
        Build logs for more information.
      CLOUD_BUILD_REQUEST_FAILED: Cloud Build failed to fulfill Cloud Deploy's
        request. See failure_message for additional details.
      DEPLOY_FEATURE_NOT_SUPPORTED: The deploy operation had a feature
        configured that is not supported.
    """
        FAILURE_CAUSE_UNSPECIFIED = 0
        CLOUD_BUILD_UNAVAILABLE = 1
        EXECUTION_FAILED = 2
        DEADLINE_EXCEEDED = 3
        MISSING_RESOURCES_FOR_CANARY = 4
        CLOUD_BUILD_REQUEST_FAILED = 5
        DEPLOY_FEATURE_NOT_SUPPORTED = 6
    artifact = _messages.MessageField('DeployArtifact', 1)
    build = _messages.StringField(2)
    failureCause = _messages.EnumField('FailureCauseValueValuesEnum', 3)
    failureMessage = _messages.StringField(4)
    metadata = _messages.MessageField('DeployJobRunMetadata', 5)