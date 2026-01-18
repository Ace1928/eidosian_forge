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
def add_attribute(self, key: str, value: Any) -> None:
    if self.attributes is None:
        self.attributes = {}
    self.attributes[key] = value