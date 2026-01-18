from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def _get_default_task_for_llm_inference_task(llm_inference_task: Optional[str]) -> Optional[str]:
    """
    Get corresponding original Transformers task for the given LLM inference task.

    NB: This assumes there is only one original Transformers task for each LLM inference
      task, which might not be true in the future.
    """
    for task, llm_tasks in _SUPPORTED_LLM_INFERENCE_TASK_TYPES_BY_PIPELINE_TASK.items():
        if llm_inference_task in llm_tasks:
            return task
    return None