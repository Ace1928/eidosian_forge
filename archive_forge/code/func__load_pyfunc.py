from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental
def _load_pyfunc(model_path: str, model_config: Optional[Dict[str, Any]]=None):
    context, chat_model, signature = _load_context_model_and_signature(model_path, model_config)
    return _ChatModelPyfuncWrapper(chat_model=chat_model, context=context, signature=signature)