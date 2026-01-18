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
def _fallback_serialize(obj: Any) -> str:
    try:
        return f'<<non-serializable: {type(obj).__qualname__}>>'
    except Exception:
        return '<<non-serializable>>'