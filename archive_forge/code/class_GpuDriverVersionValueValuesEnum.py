from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GpuDriverVersionValueValuesEnum(_messages.Enum):
    """Mode for how the GPU driver is installed.

    Values:
      GPU_DRIVER_VERSION_UNSPECIFIED: Default value is to not install any GPU
        driver.
      INSTALLATION_DISABLED: Disable GPU driver auto installation and needs
        manual installation
      DEFAULT: "Default" GPU driver in COS and Ubuntu.
      LATEST: "Latest" GPU driver in COS.
    """
    GPU_DRIVER_VERSION_UNSPECIFIED = 0
    INSTALLATION_DISABLED = 1
    DEFAULT = 2
    LATEST = 3