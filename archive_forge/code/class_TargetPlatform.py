from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class TargetPlatform(str, Enum):
    """Enum of valid target platforms."""
    linux_amd64 = 'linux/amd64'
    linux_arm64 = 'linux/arm64'