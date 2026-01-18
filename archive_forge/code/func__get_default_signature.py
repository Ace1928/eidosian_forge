import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_pip_requirements
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types.llm import (
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _get_default_signature():
    """
    Generates a default signature for the ``sentence_transformers`` flavor to be applied if not
    set or overridden by supplying the `signature` argument to `log_model` or `save_model`.
    """
    return ModelSignature(inputs=Schema([ColSpec('string')]), outputs=Schema([TensorSpec(np.dtype('float64'), [-1])]))