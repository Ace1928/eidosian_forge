from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_schema_with_defaults(self, index_schema: Optional[Union[Dict[str, ListOfDict], str, os.PathLike]]=None, vector_schema: Optional[Dict[str, Union[str, int]]]=None) -> 'RedisModel':
    from langchain_community.vectorstores.redis.schema import RedisModel, read_schema
    schema = RedisModel()
    if index_schema:
        schema_values = read_schema(index_schema)
        schema = RedisModel(**schema_values)
        schema.add_content_field()
    try:
        schema.content_vector
        if vector_schema:
            logger.warning('`vector_schema` is ignored since content_vector is ' + 'overridden in `index_schema`.')
    except ValueError:
        vector_field = self.DEFAULT_VECTOR_SCHEMA.copy()
        if vector_schema:
            vector_field.update(vector_schema)
        schema.add_vector_field(vector_field)
    return schema