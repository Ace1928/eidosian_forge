from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetRender(_messages.Message):
    """Details of rendering for a single target.

  Enums:
    FailureCauseValueValuesEnum: Output only. Reason this render failed. This
      will always be unspecified while the render in progress.
    RenderingStateValueValuesEnum: Output only. Current state of the render
      operation for this Target.

  Fields:
    failureCause: Output only. Reason this render failed. This will always be
      unspecified while the render in progress.
    failureMessage: Output only. Additional information about the render
      failure, if available.
    metadata: Output only. Metadata related to the `Release` render for this
      Target.
    renderingBuild: Output only. The resource name of the Cloud Build `Build`
      object that is used to render the manifest for this target. Format is
      `projects/{project}/locations/{location}/builds/{build}`.
    renderingState: Output only. Current state of the render operation for
      this Target.
  """

    class FailureCauseValueValuesEnum(_messages.Enum):
        """Output only. Reason this render failed. This will always be
    unspecified while the render in progress.

    Values:
      FAILURE_CAUSE_UNSPECIFIED: No reason for failure is specified.
      CLOUD_BUILD_UNAVAILABLE: Cloud Build is not available, either because it
        is not enabled or because Cloud Deploy has insufficient permissions.
        See [required permission](https://cloud.google.com/deploy/docs/cloud-
        deploy-service-account#required_permissions).
      EXECUTION_FAILED: The render operation did not complete successfully;
        check Cloud Build logs.
      CLOUD_BUILD_REQUEST_FAILED: Cloud Build failed to fulfill Cloud Deploy's
        request. See failure_message for additional details.
      VERIFICATION_CONFIG_NOT_FOUND: The render operation did not complete
        successfully because the verification stanza required for verify was
        not found on the Skaffold configuration.
      CUSTOM_ACTION_NOT_FOUND: The render operation did not complete
        successfully because the custom action required for predeploy or
        postdeploy was not found in the Skaffold configuration. See
        failure_message for additional details.
      DEPLOYMENT_STRATEGY_NOT_SUPPORTED: Release failed during rendering
        because the release configuration is not supported with the specified
        deployment strategy.
      RENDER_FEATURE_NOT_SUPPORTED: The render operation had a feature
        configured that is not supported.
    """
        FAILURE_CAUSE_UNSPECIFIED = 0
        CLOUD_BUILD_UNAVAILABLE = 1
        EXECUTION_FAILED = 2
        CLOUD_BUILD_REQUEST_FAILED = 3
        VERIFICATION_CONFIG_NOT_FOUND = 4
        CUSTOM_ACTION_NOT_FOUND = 5
        DEPLOYMENT_STRATEGY_NOT_SUPPORTED = 6
        RENDER_FEATURE_NOT_SUPPORTED = 7

    class RenderingStateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the render operation for this Target.

    Values:
      TARGET_RENDER_STATE_UNSPECIFIED: The render operation state is
        unspecified.
      SUCCEEDED: The render operation has completed successfully.
      FAILED: The render operation has failed.
      IN_PROGRESS: The render operation is in progress.
    """
        TARGET_RENDER_STATE_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2
        IN_PROGRESS = 3
    failureCause = _messages.EnumField('FailureCauseValueValuesEnum', 1)
    failureMessage = _messages.StringField(2)
    metadata = _messages.MessageField('RenderMetadata', 3)
    renderingBuild = _messages.StringField(4)
    renderingState = _messages.EnumField('RenderingStateValueValuesEnum', 5)