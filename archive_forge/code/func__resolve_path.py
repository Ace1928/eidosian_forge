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
def _resolve_path(self, index_path, filename):
    is_local = os.path.isdir(index_path)
    try:
        resolved_archive_file = cached_file(index_path, filename)
    except EnvironmentError:
        msg = f"Can't load '{filename}'. Make sure that:\n\n- '{index_path}' is a correct remote path to a directory containing a file named {filename}\n\n- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
        raise EnvironmentError(msg)
    if is_local:
        logger.info(f'loading file {resolved_archive_file}')
    else:
        logger.info(f'loading file {filename} from cache at {resolved_archive_file}')
    return resolved_archive_file