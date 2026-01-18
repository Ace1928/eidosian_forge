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
def _safe_serialize(obj: dict) -> str:
    try:
        return json.dumps(wandb.data_types._json_helper(obj, None), skipkeys=True, default=_fallback_serialize)
    except Exception:
        return '{}'