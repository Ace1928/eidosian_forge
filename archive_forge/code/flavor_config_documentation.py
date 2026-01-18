from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS
from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set

    Updates the flavor configuration that was saved with save_pretrained=False to the one that
    includes the local path to the model binary file.
    