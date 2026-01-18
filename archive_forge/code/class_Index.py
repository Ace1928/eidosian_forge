import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer
class Index:
    """
    A base class for the Indices encapsulated by the [`RagRetriever`].
    """

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (`np.ndarray` of shape `(batch_size, n_docs)`):
                A tensor of document indices.
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs` documents.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                An array of query vectors.
            n_docs (`int`):
                The number of docs retrieved per query.

        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of retrieved documents. `np.ndarray` of
            shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns `True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError