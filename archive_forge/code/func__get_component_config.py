from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS
from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set
def _get_component_config(component, key, save_pretrained=True, default_repo=None):
    conf = {FlavorKey.COMPONENT_TYPE.format(key): _get_instance_type(component)}
    if not save_pretrained:
        repo = getattr(component, 'name_or_path', default_repo)
        revision = get_latest_commit_for_repo(repo)
        conf[FlavorKey.COMPONENT_NAME.format(key)] = repo
        conf[FlavorKey.COMPONENT_REVISION.format(key)] = revision
    return conf