import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from mlflow.exceptions import MlflowException
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.utils import _get_default_model, _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import (
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
def _format_args_string(grading_context_columns: Optional[List[str]], eval_values, indx) -> str:
    import pandas as pd
    args_dict = {}
    for arg in grading_context_columns:
        if arg in eval_values:
            args_dict[arg] = eval_values[arg].iloc[indx] if isinstance(eval_values[arg], pd.Series) else eval_values[arg][indx]
        else:
            raise MlflowException(f'{arg} does not exist in the eval function {list(eval_values.keys())}.')
    return '' if args_dict is None or len(args_dict) == 0 else 'Additional information used by the model:\n' + '\n'.join([f'key: {arg}\nvalue:\n{arg_value}' for arg, arg_value in args_dict.items()])