from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def _search_with_score_and_embeddings_by_vector(self, embedding: List[float], k: int=DEFAULT_TOP_K, filter: Optional[Dict[str, Any]]=None, brute_force: bool=False, fraction_lists_to_search: Optional[float]=None) -> List[Tuple[Document, List[float], float]]:
    from google.cloud import bigquery
    if not self._have_index and (not self._creating_index):
        self._initialize_vector_index()
    filter_expr = 'TRUE'
    if filter:
        filter_expressions = []
        for i in filter.items():
            if isinstance(i[1], float):
                expr = f"ABS(CAST(JSON_VALUE(base.`{self.metadata_field}`,'$.{i[0]}') AS FLOAT64) - {i[1]}) <= {sys.float_info.epsilon}"
            else:
                val = str(i[1]).replace('"', '\\"')
                expr = f'''JSON_VALUE(base.`{self.metadata_field}`,'$.{i[0]}') = "{val}"'''
            filter_expressions.append(expr)
        filter_expression_str = ' AND '.join(filter_expressions)
        filter_expr += f' AND ({filter_expression_str})'
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter('v', 'FLOAT64', embedding)], use_query_cache=False, priority=bigquery.QueryPriority.BATCH)
    if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        distance_type = 'EUCLIDEAN'
    elif self.distance_strategy == DistanceStrategy.COSINE:
        distance_type = 'COSINE'
    else:
        distance_type = 'EUCLIDEAN'
    if brute_force:
        options_string = ',options => \'{"use_brute_force":true}\''
    elif fraction_lists_to_search:
        if fraction_lists_to_search == 0 or fraction_lists_to_search >= 1.0:
            raise ValueError('`fraction_lists_to_search` must be between 0.0 and 1.0')
        options_string = f""",options => '{{"fraction_lists_to_search":{fraction_lists_to_search}}}'"""
    else:
        options_string = ''
    query = f'\n            SELECT\n                base.*,\n                distance AS _vector_search_distance\n            FROM VECTOR_SEARCH(\n                TABLE `{self.full_table_id}`,\n                "{self.text_embedding_field}",\n                (SELECT @v AS {self.text_embedding_field}),\n                distance_type => "{distance_type}",\n                top_k => {k}\n                {options_string}\n            )\n            WHERE {filter_expr}\n            LIMIT {k}\n        '
    document_tuples: List[Tuple[Document, List[float], float]] = []
    job = self.bq_client.query(query, job_config=job_config, api_method=bigquery.enums.QueryApiMethod.QUERY)
    for row in job:
        metadata = row[self.metadata_field]
        if metadata:
            if not isinstance(metadata, dict):
                metadata = json.loads(metadata)
        else:
            metadata = {}
        metadata['__id'] = row[self.doc_id_field]
        metadata['__job_id'] = job.job_id
        doc = Document(page_content=row[self.content_field], metadata=metadata)
        document_tuples.append((doc, row[self.text_embedding_field], row['_vector_search_distance']))
    return document_tuples