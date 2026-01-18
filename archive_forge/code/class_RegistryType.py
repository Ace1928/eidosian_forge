from enum import Enum
from typing import List, Optional
from pydantic import (  # type: ignore
import wandb
from wandb.sdk.launch.utils import (
class RegistryType(str, Enum):
    """Enum of valid registry types."""
    ecr = 'ecr'
    acr = 'acr'
    gcr = 'gcr'