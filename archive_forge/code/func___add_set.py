from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def __add_set(self, collection_name: str, engine: ENGINES='FaissFlat', metric: DISTANCE_METRICS='L2') -> str:
    query = _add_descriptorset('AddDescriptorSet', collection_name, self.embedding_dimension, engine=getattr(engine, 'value', engine), metric=getattr(metric, 'value', metric))
    response, _ = self.__run_vdms_query([query])
    if 'FailedCommand' in response[0]:
        raise ValueError(f'Failed to add collection {collection_name}')
    return collection_name