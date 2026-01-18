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
def _prepare_and_validate_batches(self, texts: List[str], embeddings_type: Optional[str]=None) -> Tuple[List[List[float]], List[List[str]]]:
    """Prepares text batches with one-time validation of batch size.
        Batch size varies between GCP regions and individual project quotas.
        # Returns embeddings of the first text batch that went through,
        # and text batches for the rest of the texts.
        """
    from google.api_core.exceptions import InvalidArgument
    batches = VertexAIEmbeddings._prepare_batches(texts, self.instance['batch_size'])
    if len(batches[0]) <= self.instance['min_good_batch_size']:
        return ([], batches)
    with self.instance['lock']:
        if self.instance['batch_size_validated']:
            if len(batches[0]) <= self.instance['batch_size']:
                return ([], batches)
            else:
                return ([], VertexAIEmbeddings._prepare_batches(texts, self.instance['batch_size']))
        first_batch = batches[0]
        first_result = []
        had_failure = False
        while True:
            try:
                first_result = self._get_embeddings_with_retry(first_batch, embeddings_type)
                break
            except InvalidArgument:
                had_failure = True
                first_batch_len = len(first_batch)
                if first_batch_len == self.instance['min_batch_size']:
                    raise
                first_batch_len = max(self.instance['min_batch_size'], int(first_batch_len / 2))
                first_batch = first_batch[:first_batch_len]
        first_batch_len = len(first_batch)
        self.instance['min_good_batch_size'] = max(self.instance['min_good_batch_size'], first_batch_len)
        if had_failure or first_batch_len == self.instance['max_batch_size']:
            self.instance['batch_size'] = first_batch_len
            self.instance['batch_size_validated'] = True
            if first_batch_len != self.instance['max_batch_size']:
                batches = VertexAIEmbeddings._prepare_batches(texts[first_batch_len:], self.instance['batch_size'])
        else:
            batches = batches[1:]
    return (first_result, batches)