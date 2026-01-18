import logging
from typing import (
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
@staticmethod
def __validate_distance_strategy(distance_strategy: DistanceStrategy) -> None:
    if distance_strategy not in [DistanceStrategy.COSINE, DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.MAX_INNER_PRODUCT]:
        raise ValueError(f'Distance strategy {distance_strategy} not implemented.')