import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.data_types
from wandb.sdk.data_types import _dtypes
from wandb.sdk.data_types.base_types.media import Media
class StatusCode(str, Enum):
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'

    def __str__(self) -> str:
        return str(self.value)