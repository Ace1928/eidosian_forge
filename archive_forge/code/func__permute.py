import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
@staticmethod
def _permute(texts: List[str], sorter: Callable=len) -> Tuple[List[str], Callable]:
    """Sort texts in ascending order, and
        delivers a lambda expr, which can sort a same length list
        https://github.com/UKPLab/sentence-transformers/blob/
        c5f93f70eca933c78695c5bc686ceda59651ae3b/sentence_transformers/SentenceTransformer.py#L156

        Args:
            texts (List[str]): _description_
            sorter (Callable, optional): _description_. Defaults to len.

        Returns:
            Tuple[List[str], Callable]: _description_

        Example:
            ```
            texts = ["one","three","four"]
            perm_texts, undo = self._permute(texts)
            texts == undo(perm_texts)
            ```
        """
    if len(texts) == 1:
        return (texts, lambda t: t)
    length_sorted_idx = np.argsort([-sorter(sen) for sen in texts])
    texts_sorted = [texts[idx] for idx in length_sorted_idx]
    return (texts_sorted, lambda unsorted_embeddings: [unsorted_embeddings[idx] for idx in np.argsort(length_sorted_idx)])