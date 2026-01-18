from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class EnvironmentType(str, Enum):
    """Enum of valid environment types."""
    aws = 'aws'
    gcp = 'gcp'
    azure = 'azure'