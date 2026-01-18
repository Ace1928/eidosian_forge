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
def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
    return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]