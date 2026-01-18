import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
def _infer_schema_from_type_hint(type_hint, examples=None):
    has_examples = examples is not None
    if has_examples:
        _validate_is_list(examples)
        _validate_non_empty(examples)
    if type_hint == List[str]:
        if has_examples:
            _validate_is_all_string(examples)
        return Schema([ColSpec(type='string', name=None)])
    elif type_hint == List[Dict[str, str]]:
        if has_examples:
            _validate_dict_examples(examples)
            return Schema([ColSpec(type='string', name=name) for name in examples[0]])
        else:
            _logger.warning(f'Could not infer schema for {type_hint} because example is missing')
            return Schema([ColSpec(type='string', name=None)])
    else:
        _logger.info('Unsupported type hint: %s, skipping schema inference', type_hint)
        return None