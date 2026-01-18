import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _get_dimension(self) -> int:
    """
        Get the dimension of the vector using embedding functions.
        """
    return len(self._embedding_function.embed_query('test embedding length'))