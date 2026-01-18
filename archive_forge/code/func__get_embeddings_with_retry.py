import logging
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Literal, Optional, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.vertexai import _VertexAICommon
from langchain_community.utilities.vertexai import raise_vertex_import_error
def _get_embeddings_with_retry(self, texts: List[str], embeddings_type: Optional[str]=None) -> List[List[float]]:
    """Makes a Vertex AI model request with retry logic."""
    from google.api_core.exceptions import Aborted, DeadlineExceeded, ResourceExhausted, ServiceUnavailable
    errors = [ResourceExhausted, ServiceUnavailable, Aborted, DeadlineExceeded]
    retry_decorator = create_base_retry_decorator(error_types=errors, max_retries=self.max_retries)

    @retry_decorator
    def _completion_with_retry(texts_to_process: List[str]) -> Any:
        if embeddings_type and self.instance['embeddings_task_type_supported']:
            from vertexai.language_models import TextEmbeddingInput
            requests = [TextEmbeddingInput(text=t, task_type=embeddings_type) for t in texts_to_process]
        else:
            requests = texts_to_process
        embeddings = self.client.get_embeddings(requests)
        return [embs.values for embs in embeddings]
    return _completion_with_retry(texts)