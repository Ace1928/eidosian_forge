from abc import ABC
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _check_docarray_import() -> None:
    try:
        import docarray
        da_version = docarray.__version__.split('.')
        if int(da_version[0]) == 0 and int(da_version[1]) <= 31:
            raise ImportError(f'To use the DocArrayHnswSearch VectorStore the docarray version >=0.32.0 is expected, received: {docarray.__version__}.To upgrade, please run: `pip install -U docarray`.')
    except ImportError:
        raise ImportError('Could not import docarray python package. Please install it with `pip install "langchain[docarray]"`.')