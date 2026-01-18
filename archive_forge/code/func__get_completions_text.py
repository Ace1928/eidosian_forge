from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def _get_completions_text(prompt: str, output_tensor: List[int], pipeline):
    """Decode generated text from output tensor and remove the input prompt."""
    generated_text = pipeline.tokenizer.decode(output_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    prompt_ids_without_special_tokens = pipeline.tokenizer(prompt, return_tensors=pipeline.framework, add_special_tokens=False)['input_ids'][0]
    prompt_length = len(pipeline.tokenizer.decode(prompt_ids_without_special_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return generated_text[prompt_length:].lstrip()