from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS
from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set
def _generate_base_config(pipeline, torch_dtype=None):
    flavor_conf = {FlavorKey.TASK: pipeline.task, FlavorKey.INSTANCE_TYPE: _get_instance_type(pipeline)}
    if (framework := getattr(pipeline, 'framework', None)):
        flavor_conf[FlavorKey.FRAMEWORK] = framework
    if (torch_dtype := (torch_dtype or _extract_torch_dtype_if_set(pipeline))):
        flavor_conf[FlavorKey.TORCH_DTYPE] = str(torch_dtype)
    return flavor_conf