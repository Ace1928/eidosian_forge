from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventValueValuesEnum(_messages.Enum):
    """The event being reported.

    Values:
      UNKNOWN_EVENT: Invalid event.
      OS_START: The time the VM started.
      CONTAINER_START: Our container code starts running. Multiple containers
        could be distinguished with WorkerMessage.labels if desired.
      NETWORK_UP: The worker has a functional external network connection.
      STAGING_FILES_DOWNLOAD_START: Started downloading staging files.
      STAGING_FILES_DOWNLOAD_FINISH: Finished downloading all staging files.
      SDK_INSTALL_START: For applicable SDKs, started installation of SDK and
        worker packages.
      SDK_INSTALL_FINISH: Finished installing SDK.
    """
    UNKNOWN_EVENT = 0
    OS_START = 1
    CONTAINER_START = 2
    NETWORK_UP = 3
    STAGING_FILES_DOWNLOAD_START = 4
    STAGING_FILES_DOWNLOAD_FINISH = 5
    SDK_INSTALL_START = 6
    SDK_INSTALL_FINISH = 7