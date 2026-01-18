from __future__ import annotations
import logging
import os
import warnings
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from tenacity import (
from langchain_community.utils.openai import is_openai_v1
def _get_len_safe_embeddings(self, texts: List[str], *, engine: str, chunk_size: Optional[int]=None) -> List[List[float]]:
    """
        Generate length-safe embeddings for a list of texts.

        This method handles tokenization and embedding generation, respecting the
        set embedding context length and chunk size. It supports both tiktoken
        and HuggingFace tokenizer based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """
    tokens = []
    indices = []
    model_name = self.tiktoken_model_name or self.model
    _chunk_size = chunk_size or self.chunk_size
    if not self.tiktoken_enabled:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ValueError('Could not import transformers python package. This is needed in order to for OpenAIEmbeddings without `tiktoken`. Please install it with `pip install transformers`. ')
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        for i, text in enumerate(texts):
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            for j in range(0, len(tokenized), self.embedding_ctx_length):
                token_chunk = tokenized[j:j + self.embedding_ctx_length]
                chunk_text = tokenizer.decode(token_chunk)
                tokens.append(chunk_text)
                indices.append(i)
    else:
        try:
            import tiktoken
        except ImportError:
            raise ImportError('Could not import tiktoken python package. This is needed in order to for OpenAIEmbeddings. Please install it with `pip install tiktoken`.')
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning('Warning: model not found. Using cl100k_base encoding.')
            model = 'cl100k_base'
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            if self.model.endswith('001'):
                text = text.replace('\n', ' ')
            token = encoding.encode(text=text, allowed_special=self.allowed_special, disallowed_special=self.disallowed_special)
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens.append(token[j:j + self.embedding_ctx_length])
                indices.append(i)
    if self.show_progress_bar:
        try:
            from tqdm.auto import tqdm
            _iter = tqdm(range(0, len(tokens), _chunk_size))
        except ImportError:
            _iter = range(0, len(tokens), _chunk_size)
    else:
        _iter = range(0, len(tokens), _chunk_size)
    batched_embeddings: List[List[float]] = []
    for i in _iter:
        response = embed_with_retry(self, input=tokens[i:i + _chunk_size], **self._invocation_params)
        if not isinstance(response, dict):
            response = response.dict()
        batched_embeddings.extend((r['embedding'] for r in response['data']))
    results: List[List[List[float]]] = [[] for _ in range(len(texts))]
    num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
    for i in range(len(indices)):
        if self.skip_empty and len(batched_embeddings[i]) == 1:
            continue
        results[indices[i]].append(batched_embeddings[i])
        num_tokens_in_batch[indices[i]].append(len(tokens[i]))
    embeddings: List[List[float]] = [[] for _ in range(len(texts))]
    for i in range(len(texts)):
        _result = results[i]
        if len(_result) == 0:
            average_embedded = embed_with_retry(self, input='', **self._invocation_params)
            if not isinstance(average_embedded, dict):
                average_embedded = average_embedded.dict()
            average = average_embedded['data'][0]['embedding']
        else:
            average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
        embeddings[i] = (average / np.linalg.norm(average)).tolist()
    return embeddings