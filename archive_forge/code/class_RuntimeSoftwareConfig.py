from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeSoftwareConfig(_messages.Message):
    """Specifies the selection and configuration of software inside the
  runtime. The properties to set on runtime. Properties keys are specified in
  `key:value` format, for example: * `idle_shutdown: true` *
  `idle_shutdown_timeout: 180` * `enable_health_monitoring: true`

  Enums:
    PostStartupScriptBehaviorValueValuesEnum: Behavior for the post startup
      script.

  Fields:
    customGpuDriverPath: Specify a custom Cloud Storage path where the GPU
      driver is stored. If not specified, we'll automatically choose from
      official GPU drivers.
    disableTerminal: Bool indicating whether JupyterLab terminal will be
      available or not. Default: False
    enableHealthMonitoring: Verifies core internal services are running.
      Default: True
    idleShutdown: Runtime will automatically shutdown after
      idle_shutdown_time. Default: True
    idleShutdownTimeout: Time in minutes to wait before shutting down runtime.
      Default: 180 minutes
    installGpuDriver: Install Nvidia Driver automatically. Default: True
    kernels: Optional. Use a list of container images to use as Kernels in the
      notebook instance.
    mixerDisabled: Bool indicating whether mixer client should be disabled.
      Default: False
    notebookUpgradeSchedule: Cron expression in UTC timezone, used to schedule
      instance auto upgrade. Please follow the [cron
      format](https://en.wikipedia.org/wiki/Cron).
    postStartupScript: Path to a Bash script that automatically runs after a
      notebook instance fully boots up. The path must be a URL or Cloud
      Storage path (`gs://path-to-file/file-name`).
    postStartupScriptBehavior: Behavior for the post startup script.
    upgradeable: Output only. Bool indicating whether an newer image is
      available in an image family.
    version: Output only. version of boot image such as M100, from release
      label of the image.
  """

    class PostStartupScriptBehaviorValueValuesEnum(_messages.Enum):
        """Behavior for the post startup script.

    Values:
      POST_STARTUP_SCRIPT_BEHAVIOR_UNSPECIFIED: Unspecified post startup
        script behavior. Will run only once at creation.
      RUN_EVERY_START: Runs the post startup script provided during creation
        at every start.
      DOWNLOAD_AND_RUN_EVERY_START: Downloads and runs the provided post
        startup script at every start.
    """
        POST_STARTUP_SCRIPT_BEHAVIOR_UNSPECIFIED = 0
        RUN_EVERY_START = 1
        DOWNLOAD_AND_RUN_EVERY_START = 2
    customGpuDriverPath = _messages.StringField(1)
    disableTerminal = _messages.BooleanField(2)
    enableHealthMonitoring = _messages.BooleanField(3)
    idleShutdown = _messages.BooleanField(4)
    idleShutdownTimeout = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    installGpuDriver = _messages.BooleanField(6)
    kernels = _messages.MessageField('ContainerImage', 7, repeated=True)
    mixerDisabled = _messages.BooleanField(8)
    notebookUpgradeSchedule = _messages.StringField(9)
    postStartupScript = _messages.StringField(10)
    postStartupScriptBehavior = _messages.EnumField('PostStartupScriptBehaviorValueValuesEnum', 11)
    upgradeable = _messages.BooleanField(12)
    version = _messages.StringField(13)