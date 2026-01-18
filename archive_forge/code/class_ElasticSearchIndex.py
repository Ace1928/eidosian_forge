import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
class ElasticSearchIndex(BaseIndex):
    """
    Sparse index using Elasticsearch. It is used to index text and run queries based on BM25 similarity.
    An Elasticsearch server needs to be accessible, and a python client is declared with
    ```
    es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    ```
    for example.
    """

    def __init__(self, host: Optional[str]=None, port: Optional[int]=None, es_client: Optional['Elasticsearch']=None, es_index_name: Optional[str]=None, es_index_config: Optional[dict]=None):
        if not _has_elasticsearch:
            raise ImportError('You must install ElasticSearch to use ElasticSearchIndex. To do so you can run `pip install elasticsearch==7.7.1 for example`')
        if es_client is not None and (host is not None or port is not None):
            raise ValueError('Please specify either `es_client` or `(host, port)`, but not both.')
        host = host or 'localhost'
        port = port or 9200
        import elasticsearch.helpers
        from elasticsearch import Elasticsearch
        self.es_client = es_client if es_client is not None else Elasticsearch([{'host': host, 'port': str(port)}])
        self.es_index_name = es_index_name if es_index_name is not None else 'huggingface_datasets_' + os.path.basename(tempfile.NamedTemporaryFile().name)
        self.es_index_config = es_index_config if es_index_config is not None else {'settings': {'number_of_shards': 1, 'analysis': {'analyzer': {'stop_standard': {'type': 'standard', ' stopwords': '_english_'}}}}, 'mappings': {'properties': {'text': {'type': 'text', 'analyzer': 'standard', 'similarity': 'BM25'}}}}

    def add_documents(self, documents: Union[List[str], 'Dataset'], column: Optional[str]=None):
        """
        Add documents to the index.
        If the documents are inside a certain column, you can specify it using the `column` argument.
        """
        index_name = self.es_index_name
        index_config = self.es_index_config
        self.es_client.indices.create(index=index_name, body=index_config)
        number_of_docs = len(documents)
        progress = hf_tqdm(unit='docs', total=number_of_docs)
        successes = 0

        def passage_generator():
            if column is not None:
                for i, example in enumerate(documents):
                    yield {'text': example[column], '_id': i}
            else:
                for i, example in enumerate(documents):
                    yield {'text': example, '_id': i}
        import elasticsearch as es
        for ok, action in es.helpers.streaming_bulk(client=self.es_client, index=index_name, actions=passage_generator()):
            progress.update(1)
            successes += ok
        if successes != len(documents):
            logger.warning(f'Some documents failed to be added to ElasticSearch. Failures: {len(documents) - successes}/{len(documents)}')
        logger.info(f'Indexed {successes:d} documents')

    def search(self, query: str, k=10, **kwargs) -> SearchResults:
        """Find the nearest examples indices to the query.

        Args:
            query (`str`): The query as a string.
            k (`int`): The number of examples to retrieve.

        Ouput:
            scores (`List[List[float]`): The retrieval scores of the retrieved examples.
            indices (`List[List[int]]`): The indices of the retrieved examples.
        """
        response = self.es_client.search(index=self.es_index_name, body={'query': {'multi_match': {'query': query, 'fields': ['text'], 'type': 'cross_fields'}}, 'size': k}, **kwargs)
        hits = response['hits']['hits']
        return SearchResults([hit['_score'] for hit in hits], [int(hit['_id']) for hit in hits])

    def search_batch(self, queries, k: int=10, max_workers=10, **kwargs) -> BatchedSearchResults:
        import concurrent.futures
        total_scores, total_indices = ([None] * len(queries), [None] * len(queries))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.search, query, k, **kwargs): i for i, query in enumerate(queries)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                results: SearchResults = future.result()
                total_scores[index] = results.scores
                total_indices[index] = results.indices
        return BatchedSearchResults(total_indices=total_indices, total_scores=total_scores)