import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
@dataclass
class UsageMetrics:
    elapsed_time: float = None
    prompt_tokens: int = None
    completion_tokens: int = None
    total_tokens: int = None