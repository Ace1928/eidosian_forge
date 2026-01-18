from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def _get_slow_token_ids(seq: str):
    return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(seq))