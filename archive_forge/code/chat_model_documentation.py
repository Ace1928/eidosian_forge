from typing import Any, Dict, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental

        Args:
            model_input: Model input data in the form of a chat request.
            params: Additional parameters to pass to the model for inference.
                       Unused in this implementation, as the params are handled
                       via ``self._convert_input()``.

        Returns:
            Model predictions in :py:class:`~ChatResponse` format.
        