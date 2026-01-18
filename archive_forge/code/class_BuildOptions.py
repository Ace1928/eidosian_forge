from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildOptions(_messages.Message):
    """Optional arguments to enable specific features of builds.

  Enums:
    LogStreamingOptionValueValuesEnum: Option to define build log streaming
      behavior to Google Cloud Storage.
    LoggingValueValuesEnum: Option to specify the logging mode, which
      determines if and where build logs are stored.
    MachineTypeValueValuesEnum: Compute Engine machine type on which to run
      the build.
    RequestedVerifyOptionValueValuesEnum: Requested verifiability options.
    SourceProvenanceHashValueListEntryValuesEnum:
    SubstitutionOptionValueValuesEnum: Option to specify behavior when there
      is an error in the substitution checks. NOTE: this is always set to
      ALLOW_LOOSE for triggered builds and cannot be overridden in the build
      configuration file.

  Fields:
    anthosCluster: Details about how this build should be executed on a Anthos
      cluster.
    cluster: Details about how this build should be executed on a GKE cluster.
    diskSizeGb: Requested disk size for the VM that runs the build. Note that
      this is *NOT* "disk free"; some of the space will be used by the
      operating system and build utilities. Also note that this is the minimum
      disk size that will be allocated for the build -- the build may run with
      a larger disk than requested. At present, the maximum disk size is
      2000GB; builds that request more than the maximum are rejected with an
      error.
    dynamicSubstitutions: Option to specify whether or not to apply bash style
      string operations to the substitutions. NOTE: this is always enabled for
      triggered builds and cannot be overridden in the build configuration
      file.
    env: A list of global environment variable definitions that will exist for
      all build steps in this build. If a variable is defined in both globally
      and in a build step, the variable will use the build step value. The
      elements are of the form "KEY=VALUE" for the environment variable "KEY"
      being given the value "VALUE".
    logStreamingOption: Option to define build log streaming behavior to
      Google Cloud Storage.
    logging: Option to specify the logging mode, which determines if and where
      build logs are stored.
    machineType: Compute Engine machine type on which to run the build.
    pool: Optional. Specification for execution on a `WorkerPool`. See
      [running builds in a private
      pool](https://cloud.google.com/build/docs/private-pools/run-builds-in-
      private-pool) for more information.
    requestedVerifyOption: Requested verifiability options.
    secretEnv: A list of global environment variables, which are encrypted
      using a Cloud Key Management Service crypto key. These values must be
      specified in the build's `Secret`. These variables will be available to
      all build steps in this build.
    sourceProvenanceHash: Requested hash for SourceProvenance.
    substitutionOption: Option to specify behavior when there is an error in
      the substitution checks. NOTE: this is always set to ALLOW_LOOSE for
      triggered builds and cannot be overridden in the build configuration
      file.
    volumes: Global list of volumes to mount for ALL build steps Each volume
      is created as an empty volume prior to starting the build process. Upon
      completion of the build, volumes and their contents are discarded.
      Global volume names and paths cannot conflict with the volumes defined a
      build step. Using a global volume in a build with only one step is not
      valid as it is indicative of a build request with an incorrect
      configuration.
    workerPool: This field deprecated; please use `pool.name` instead.
  """

    class LogStreamingOptionValueValuesEnum(_messages.Enum):
        """Option to define build log streaming behavior to Google Cloud Storage.

    Values:
      STREAM_DEFAULT: Service may automatically determine build log streaming
        behavior.
      STREAM_ON: Build logs should be streamed to Google Cloud Storage.
      STREAM_OFF: Build logs should not be streamed to Google Cloud Storage;
        they will be written when the build is completed.
    """
        STREAM_DEFAULT = 0
        STREAM_ON = 1
        STREAM_OFF = 2

    class LoggingValueValuesEnum(_messages.Enum):
        """Option to specify the logging mode, which determines if and where
    build logs are stored.

    Values:
      LOGGING_UNSPECIFIED: The service determines the logging mode. The
        default is `LEGACY`. Do not rely on the default logging behavior as it
        may change in the future.
      LEGACY: Build logs are stored in Cloud Logging and Cloud Storage.
      GCS_ONLY: Build logs are stored in Cloud Storage.
      STACKDRIVER_ONLY: This option is the same as CLOUD_LOGGING_ONLY.
      CLOUD_LOGGING_ONLY: Build logs are stored in Cloud Logging. Selecting
        this option will not allow [logs
        streaming](https://cloud.google.com/sdk/gcloud/reference/builds/log).
      NONE: Turn off all logging. No build logs will be captured.
    """
        LOGGING_UNSPECIFIED = 0
        LEGACY = 1
        GCS_ONLY = 2
        STACKDRIVER_ONLY = 3
        CLOUD_LOGGING_ONLY = 4
        NONE = 5

    class MachineTypeValueValuesEnum(_messages.Enum):
        """Compute Engine machine type on which to run the build.

    Values:
      UNSPECIFIED: Standard machine type.
      N1_HIGHCPU_8: Highcpu machine with 8 CPUs.
      N1_HIGHCPU_32: Highcpu machine with 32 CPUs.
      E2_HIGHCPU_8: Highcpu e2 machine with 8 CPUs.
      E2_HIGHCPU_32: Highcpu e2 machine with 32 CPUs.
    """
        UNSPECIFIED = 0
        N1_HIGHCPU_8 = 1
        N1_HIGHCPU_32 = 2
        E2_HIGHCPU_8 = 3
        E2_HIGHCPU_32 = 4

    class RequestedVerifyOptionValueValuesEnum(_messages.Enum):
        """Requested verifiability options.

    Values:
      NOT_VERIFIED: Not a verifiable build (the default).
      VERIFIED: Build must be verified.
    """
        NOT_VERIFIED = 0
        VERIFIED = 1

    class SourceProvenanceHashValueListEntryValuesEnum(_messages.Enum):
        """SourceProvenanceHashValueListEntryValuesEnum enum type.

    Values:
      NONE: No hash requested.
      SHA256: Use a sha256 hash.
      MD5: Use a md5 hash.
    """
        NONE = 0
        SHA256 = 1
        MD5 = 2

    class SubstitutionOptionValueValuesEnum(_messages.Enum):
        """Option to specify behavior when there is an error in the substitution
    checks. NOTE: this is always set to ALLOW_LOOSE for triggered builds and
    cannot be overridden in the build configuration file.

    Values:
      MUST_MATCH: Fails the build if error in substitutions checks, like
        missing a substitution in the template or in the map.
      ALLOW_LOOSE: Do not fail the build if error in substitutions checks.
    """
        MUST_MATCH = 0
        ALLOW_LOOSE = 1
    anthosCluster = _messages.MessageField('AnthosWorkerPool', 1)
    cluster = _messages.MessageField('ClusterOptions', 2)
    diskSizeGb = _messages.IntegerField(3)
    dynamicSubstitutions = _messages.BooleanField(4)
    env = _messages.StringField(5, repeated=True)
    logStreamingOption = _messages.EnumField('LogStreamingOptionValueValuesEnum', 6)
    logging = _messages.EnumField('LoggingValueValuesEnum', 7)
    machineType = _messages.EnumField('MachineTypeValueValuesEnum', 8)
    pool = _messages.MessageField('PoolOption', 9)
    requestedVerifyOption = _messages.EnumField('RequestedVerifyOptionValueValuesEnum', 10)
    secretEnv = _messages.StringField(11, repeated=True)
    sourceProvenanceHash = _messages.EnumField('SourceProvenanceHashValueListEntryValuesEnum', 12, repeated=True)
    substitutionOption = _messages.EnumField('SubstitutionOptionValueValuesEnum', 13)
    volumes = _messages.MessageField('Volume', 14, repeated=True)
    workerPool = _messages.StringField(15)