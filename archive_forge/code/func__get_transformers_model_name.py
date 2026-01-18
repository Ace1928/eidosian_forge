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
def _get_transformers_model_name(model_name_or_path):
    """
    Extract the Transformers model name from name_or_path attribute of a Transformer model.

    Normally the name_or_path attribute just points to the model name, but in Sentence
    Transformers < 2.3.0, the library loads the Transformers model after local snapshot
    download, so the name_or_path attribute points to the local filepath.
    https://github.com/UKPLab/sentence-transformers/commit/9db0f205adcf315d16961fea7e9e6906cb950d43
    """
    if (m := _LOCAL_SNAPSHOT_PATH_PATTERN.search(model_name_or_path)):
        return f'{m.group(1)}/{m.group(2)}'
    return model_name_or_path