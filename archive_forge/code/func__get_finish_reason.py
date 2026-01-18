from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def _get_finish_reason(total_tokens: int, completion_tokens: int, model_config):
    """Determine the reason that the text generation finished."""
    finish_reason = 'stop'
    if total_tokens > model_config.get('max_length', float('inf')) or completion_tokens == model_config.get('max_new_tokens', float('inf')):
        finish_reason = 'length'
    return finish_reason