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
def __update_properties(self, collection_name: str, current_collection_properties: List, new_collection_properties: Optional[List]) -> None:
    if new_collection_properties is not None:
        old_collection_properties = deepcopy(current_collection_properties)
        for prop in new_collection_properties:
            if prop not in current_collection_properties:
                current_collection_properties.append(prop)
        if current_collection_properties != old_collection_properties:
            all_queries, blob_arr = _build_property_query(collection_name, command_type='update', all_properties=current_collection_properties)
            response, _ = self.__run_vdms_query(all_queries, [blob_arr])