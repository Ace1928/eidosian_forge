from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def _get_search_client(endpoint: str, key: str, index_name: str, semantic_configuration_name: Optional[str]=None, fields: Optional[List[SearchField]]=None, vector_search: Optional[VectorSearch]=None, semantic_configurations: Optional[Union[SemanticConfiguration, List[SemanticConfiguration]]]=None, scoring_profiles: Optional[List[ScoringProfile]]=None, default_scoring_profile: Optional[str]=None, default_fields: Optional[List[SearchField]]=None, user_agent: Optional[str]='langchain', cors_options: Optional[CorsOptions]=None) -> SearchClient:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import ExhaustiveKnnAlgorithmConfiguration, ExhaustiveKnnParameters, HnswAlgorithmConfiguration, HnswParameters, SearchIndex, SemanticConfiguration, SemanticField, SemanticPrioritizedFields, SemanticSearch, VectorSearch, VectorSearchAlgorithmKind, VectorSearchAlgorithmMetric, VectorSearchProfile
    default_fields = default_fields or []
    if key is None:
        credential = DefaultAzureCredential()
    elif key.upper() == 'INTERACTIVE':
        credential = InteractiveBrowserCredential()
        credential.get_token('https://search.azure.com/.default')
    else:
        credential = AzureKeyCredential(key)
    index_client: SearchIndexClient = SearchIndexClient(endpoint=endpoint, credential=credential, user_agent=user_agent)
    try:
        index_client.get_index(name=index_name)
    except ResourceNotFoundError:
        if fields is not None:
            fields_types = {f.name: f.type for f in fields}
            mandatory_fields = {df.name: df.type for df in default_fields}
            missing_fields = {key: mandatory_fields[key] for key, value in set(mandatory_fields.items()) - set(fields_types.items())}
            if len(missing_fields) > 0:

                def fmt_err(x: str) -> str:
                    return f"{x} current type: '{fields_types.get(x, 'MISSING')}'. It has to be '{mandatory_fields.get(x)}' or you can point to a different '{mandatory_fields.get(x)}' field name by using the env variable 'AZURESEARCH_FIELDS_{x.upper()}'"
                error = '\n'.join([fmt_err(x) for x in missing_fields])
                raise ValueError(f'You need to specify at least the following fields {missing_fields} or provide alternative field names in the env variables.\n\n{error}')
        else:
            fields = default_fields
        if vector_search is None:
            vector_search = VectorSearch(algorithms=[HnswAlgorithmConfiguration(name='default', kind=VectorSearchAlgorithmKind.HNSW, parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric=VectorSearchAlgorithmMetric.COSINE)), ExhaustiveKnnAlgorithmConfiguration(name='default_exhaustive_knn', kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN, parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE))], profiles=[VectorSearchProfile(name='myHnswProfile', algorithm_configuration_name='default'), VectorSearchProfile(name='myExhaustiveKnnProfile', algorithm_configuration_name='default_exhaustive_knn')])
        if semantic_configurations:
            if not isinstance(semantic_configurations, list):
                semantic_configurations = [semantic_configurations]
            semantic_search = SemanticSearch(configurations=semantic_configurations, default_configuration_name=semantic_configuration_name)
        elif semantic_configuration_name:
            semantic_configuration = SemanticConfiguration(name=semantic_configuration_name, prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name=FIELDS_CONTENT)]))
            semantic_search = SemanticSearch(configurations=[semantic_configuration])
        else:
            semantic_search = None
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search, scoring_profiles=scoring_profiles, default_scoring_profile=default_scoring_profile, cors_options=cors_options)
        index_client.create_index(index)
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential, user_agent=user_agent)