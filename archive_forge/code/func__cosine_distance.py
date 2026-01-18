from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
from langchain.utils.math import cosine_similarity
@staticmethod
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the cosine distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.ndarray: The cosine distance.
        """
    return 1.0 - cosine_similarity(a, b)