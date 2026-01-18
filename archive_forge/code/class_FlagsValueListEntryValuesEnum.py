from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlagsValueListEntryValuesEnum(_messages.Enum):
    """FlagsValueListEntryValuesEnum enum type.

    Values:
      FLAG_UNSPECIFIED: Unspecified flag.
      IGNORE_EXIT_STATUS: Normally, a non-zero exit status causes the pipeline
        to fail. This flag allows execution of other actions to continue
        instead.
      RUN_IN_BACKGROUND: This flag allows an action to continue running in the
        background while executing subsequent actions. This is useful to
        provide services to other actions (or to provide debugging support
        tools like SSH servers).
      ALWAYS_RUN: By default, after an action fails, no further actions are
        run. This flag indicates that this action must be run even if the
        pipeline has already failed. This is useful for actions that copy
        output files off of the VM or for debugging. Note that no actions will
        be run if image prefetching fails.
      ENABLE_FUSE: Enable access to the FUSE device for this action.
        Filesystems can then be mounted into disks shared with other actions.
        The other actions do not need the `ENABLE_FUSE` flag to access the
        mounted filesystem. This has the effect of causing the container to be
        executed with `CAP_SYS_ADMIN` and exposes `/dev/fuse` to the
        container, so use it only for containers you trust.
      PUBLISH_EXPOSED_PORTS: Exposes all ports specified by `EXPOSE`
        statements in the container. To discover the host side port numbers,
        consult the `ACTION_STARTED` event in the operation metadata.
      DISABLE_IMAGE_PREFETCH: All container images are typically downloaded
        before any actions are executed. This helps prevent typos in URIs or
        issues like lack of disk space from wasting large amounts of compute
        resources. If set, this flag prevents the worker from downloading the
        image until just before the action is executed.
      DISABLE_STANDARD_ERROR_CAPTURE: A small portion of the container's
        standard error stream is typically captured and returned inside the
        `ContainerStoppedEvent`. Setting this flag disables this
        functionality.
      BLOCK_EXTERNAL_NETWORK: Prevents the container from accessing the
        external network.
    """
    FLAG_UNSPECIFIED = 0
    IGNORE_EXIT_STATUS = 1
    RUN_IN_BACKGROUND = 2
    ALWAYS_RUN = 3
    ENABLE_FUSE = 4
    PUBLISH_EXPOSED_PORTS = 5
    DISABLE_IMAGE_PREFETCH = 6
    DISABLE_STANDARD_ERROR_CAPTURE = 7
    BLOCK_EXTERNAL_NETWORK = 8