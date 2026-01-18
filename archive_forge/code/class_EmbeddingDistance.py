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
class EmbeddingDistance(str, Enum):
    """Embedding Distance Metric.

    Attributes:
        COSINE: Cosine distance metric.
        EUCLIDEAN: Euclidean distance metric.
        MANHATTAN: Manhattan distance metric.
        CHEBYSHEV: Chebyshev distance metric.
        HAMMING: Hamming distance metric.
    """
    COSINE = 'cosine'
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    CHEBYSHEV = 'chebyshev'
    HAMMING = 'hamming'