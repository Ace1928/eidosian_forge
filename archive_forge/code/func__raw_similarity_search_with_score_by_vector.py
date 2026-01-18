from __future__ import annotations
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _raw_similarity_search_with_score_by_vector(self, embedding: List[float], k: int=4, **kwargs: Any) -> List[dict]:
    """Return raw opensearch documents (dict) including vectors,
        scores most similar to the embedding vector.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of dict with its scores most similar to the embedding.

        Optional Args:
            same as `similarity_search`
        """
    search_type = kwargs.get('search_type', 'approximate_search')
    vector_field = kwargs.get('vector_field', 'vector_field')
    index_name = kwargs.get('index_name', self.index_name)
    filter = kwargs.get('filter', {})
    if self.is_aoss and search_type != 'approximate_search' and (search_type != SCRIPT_SCORING_SEARCH):
        raise ValueError('Amazon OpenSearch Service Serverless only supports `approximate_search` and `script_scoring`')
    if search_type == 'approximate_search':
        boolean_filter = kwargs.get('boolean_filter', {})
        subquery_clause = kwargs.get('subquery_clause', 'must')
        efficient_filter = kwargs.get('efficient_filter', {})
        lucene_filter = kwargs.get('lucene_filter', {})
        if boolean_filter != {} and efficient_filter != {}:
            raise ValueError('Both `boolean_filter` and `efficient_filter` are provided which is invalid')
        if lucene_filter != {} and efficient_filter != {}:
            raise ValueError('Both `lucene_filter` and `efficient_filter` are provided which is invalid. `lucene_filter` is deprecated')
        if lucene_filter != {} and boolean_filter != {}:
            raise ValueError('Both `lucene_filter` and `boolean_filter` are provided which is invalid. `lucene_filter` is deprecated')
        if efficient_filter == {} and boolean_filter == {} and (lucene_filter == {}) and (filter != {}):
            if self.engine in ['faiss', 'lucene']:
                efficient_filter = filter
            else:
                boolean_filter = filter
        if boolean_filter != {}:
            search_query = _approximate_search_query_with_boolean_filter(embedding, boolean_filter, k=k, vector_field=vector_field, subquery_clause=subquery_clause)
        elif efficient_filter != {}:
            search_query = _approximate_search_query_with_efficient_filter(embedding, efficient_filter, k=k, vector_field=vector_field)
        elif lucene_filter != {}:
            warnings.warn('`lucene_filter` is deprecated. Please use the keyword argument `efficient_filter`')
            search_query = _approximate_search_query_with_efficient_filter(embedding, lucene_filter, k=k, vector_field=vector_field)
        else:
            search_query = _default_approximate_search_query(embedding, k=k, vector_field=vector_field)
    elif search_type == SCRIPT_SCORING_SEARCH:
        space_type = kwargs.get('space_type', 'l2')
        pre_filter = kwargs.get('pre_filter', MATCH_ALL_QUERY)
        search_query = _default_script_query(embedding, k, space_type, pre_filter, vector_field)
    elif search_type == PAINLESS_SCRIPTING_SEARCH:
        space_type = kwargs.get('space_type', 'l2Squared')
        pre_filter = kwargs.get('pre_filter', MATCH_ALL_QUERY)
        search_query = _default_painless_scripting_query(embedding, k, space_type, pre_filter, vector_field)
    else:
        raise ValueError('Invalid `search_type` provided as an argument')
    response = self.client.search(index=index_name, body=search_query)
    return [hit for hit in response['hits']['hits']]