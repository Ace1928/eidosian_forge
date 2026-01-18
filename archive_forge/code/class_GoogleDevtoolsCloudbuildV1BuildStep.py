from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1BuildStep(_messages.Message):
    """A step in the build pipeline.

  Enums:
    StatusValueValuesEnum: Output only. Status of the build step. At this
      time, build step status is only updated on build completion; step status
      is not updated in real-time as the build progresses.

  Fields:
    allowExitCodes: Allow this build step to fail without failing the entire
      build if and only if the exit code is one of the specified codes. If
      allow_failure is also specified, this field will take precedence.
    allowFailure: Allow this build step to fail without failing the entire
      build. If false, the entire build will fail if this step fails.
      Otherwise, the build will succeed, but this step will still have a
      failure status. Error information will be reported in the failure_detail
      field.
    args: A list of arguments that will be presented to the step when it is
      started. If the image used to run the step's container has an
      entrypoint, the `args` are used as arguments to that entrypoint. If the
      image does not define an entrypoint, the first element in args is used
      as the entrypoint, and the remainder will be used as arguments.
    automapSubstitutions: Option to include built-in and custom substitutions
      as env variables for this build step. This option will override the
      global option in BuildOption.
    dir: Working directory to use when running this step's container. If this
      value is a relative path, it is relative to the build's working
      directory. If this value is absolute, it may be outside the build's
      working directory, in which case the contents of the path may not be
      persisted across build step executions, unless a `volume` for that path
      is specified. If the build specifies a `RepoSource` with `dir` and a
      step with a `dir`, which specifies an absolute path, the `RepoSource`
      `dir` is ignored for the step's execution.
    entrypoint: Entrypoint to be used instead of the build step image's
      default entrypoint. If unset, the image's default entrypoint is used.
    env: A list of environment variable definitions to be used when running a
      step. The elements are of the form "KEY=VALUE" for the environment
      variable "KEY" being given the value "VALUE".
    exitCode: Output only. Return code from running the step.
    id: Unique identifier for this build step, used in `wait_for` to reference
      this build step as a dependency.
    name: Required. The name of the container image that will run this
      particular build step. If the image is available in the host's Docker
      daemon's cache, it will be run directly. If not, the host will attempt
      to pull the image first, using the builder service account's credentials
      if necessary. The Docker daemon's cache will already have the latest
      versions of all of the officially supported build steps
      ([https://github.com/GoogleCloudPlatform/cloud-
      builders](https://github.com/GoogleCloudPlatform/cloud-builders)). The
      Docker daemon will also have cached many of the layers for some popular
      images, like "ubuntu", "debian", but they will be refreshed at the time
      you attempt to use them. If you built an image in a previous build step,
      it will be stored in the host's Docker daemon's cache and is available
      to use as the name for a later build step.
    pullTiming: Output only. Stores timing information for pulling this build
      step's builder image only.
    script: A shell script to be executed in the step. When script is
      provided, the user cannot specify the entrypoint or args.
    secretEnv: A list of environment variables which are encrypted using a
      Cloud Key Management Service crypto key. These values must be specified
      in the build's `Secret`.
    status: Output only. Status of the build step. At this time, build step
      status is only updated on build completion; step status is not updated
      in real-time as the build progresses.
    timeout: Time limit for executing this build step. If not defined, the
      step has no time limit and will be allowed to continue to run until
      either it completes or the build itself times out.
    timing: Output only. Stores timing information for executing this build
      step.
    volumes: List of volumes to mount into the build step. Each volume is
      created as an empty volume prior to execution of the build step. Upon
      completion of the build, volumes and their contents are discarded. Using
      a named volume in only one step is not valid as it is indicative of a
      build request with an incorrect configuration.
    waitFor: The ID(s) of the step(s) that this build step depends on. This
      build step will not start until all the build steps in `wait_for` have
      completed successfully. If `wait_for` is empty, this build step will
      start when all previous build steps in the `Build.Steps` list have
      completed successfully.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Output only. Status of the build step. At this time, build step status
    is only updated on build completion; step status is not updated in real-
    time as the build progresses.

    Values:
      STATUS_UNKNOWN: Status of the build is unknown.
      PENDING: Build has been created and is pending execution and queuing. It
        has not been queued.
      QUEUED: Build or step is queued; work has not yet begun.
      WORKING: Build or step is being executed.
      SUCCESS: Build or step finished successfully.
      FAILURE: Build or step failed to complete successfully.
      INTERNAL_ERROR: Build or step failed due to an internal cause.
      TIMEOUT: Build or step took longer than was allowed.
      CANCELLED: Build or step was canceled by a user.
      EXPIRED: Build was enqueued for longer than the value of `queue_ttl`.
    """
        STATUS_UNKNOWN = 0
        PENDING = 1
        QUEUED = 2
        WORKING = 3
        SUCCESS = 4
        FAILURE = 5
        INTERNAL_ERROR = 6
        TIMEOUT = 7
        CANCELLED = 8
        EXPIRED = 9
    allowExitCodes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    allowFailure = _messages.BooleanField(2)
    args = _messages.StringField(3, repeated=True)
    automapSubstitutions = _messages.BooleanField(4)
    dir = _messages.StringField(5)
    entrypoint = _messages.StringField(6)
    env = _messages.StringField(7, repeated=True)
    exitCode = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    id = _messages.StringField(9)
    name = _messages.StringField(10)
    pullTiming = _messages.MessageField('GoogleDevtoolsCloudbuildV1TimeSpan', 11)
    script = _messages.StringField(12)
    secretEnv = _messages.StringField(13, repeated=True)
    status = _messages.EnumField('StatusValueValuesEnum', 14)
    timeout = _messages.StringField(15)
    timing = _messages.MessageField('GoogleDevtoolsCloudbuildV1TimeSpan', 16)
    volumes = _messages.MessageField('GoogleDevtoolsCloudbuildV1Volume', 17, repeated=True)
    waitFor = _messages.StringField(18, repeated=True)