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
def _check_descriptor_exists_by_id(client: vdms.vdms, setname: str, id: str) -> Tuple[bool, Any]:
    constraints = {'id': ['==', id]}
    findDescriptor = _add_descriptor('FindDescriptor', setname, constraints=constraints, results={'list': ['id'], 'count': ''})
    all_queries = [findDescriptor]
    res, _ = client.query(all_queries)
    valid_res = _check_valid_response(all_queries, res)
    return (valid_res, findDescriptor)