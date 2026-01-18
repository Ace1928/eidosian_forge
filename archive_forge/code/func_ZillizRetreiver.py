import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.zilliz import Zilliz
def ZillizRetreiver(*args: Any, **kwargs: Any) -> ZillizRetriever:
    """Deprecated ZillizRetreiver.

    Please use ZillizRetriever ('i' before 'e') instead.

    Args:
        *args:
        **kwargs:

    Returns:
        ZillizRetriever
    """
    warnings.warn("ZillizRetreiver will be deprecated in the future. Please use ZillizRetriever ('i' before 'e') instead.", DeprecationWarning)
    return ZillizRetriever(*args, **kwargs)