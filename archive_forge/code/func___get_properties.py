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
def __get_properties(self, collection_name: str, unique_entity: Optional[bool]=False, deletion: Optional[bool]=False) -> List[str]:
    find_query = _find_property_entity(collection_name, unique_entity=unique_entity, deletion=deletion)
    response, response_blob = self.__run_vdms_query([find_query])
    if len(response_blob) > 0:
        collection_properties = _bytes2str(response_blob[0]).split(',')
    else:
        collection_properties = deepcopy(DEFAULT_PROPERTIES)
    return collection_properties