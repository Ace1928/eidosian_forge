import json
import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
class _VectorStoreClient:

    def __init__(self, host: Optional[str]=None, port: Optional[int]=None, url: Optional[str]=None):
        """
        A client you can use to query :py:class:`VectorStoreServer`.

        Please provide aither the `url`, or `host` and `port`.

        Args:
            - host: host on which `:py:class:`VectorStoreServer` listens
            - port: port on which `:py:class:`VectorStoreServer` listens
            - url: url at which `:py:class:`VectorStoreServer` listens
        """
        err = 'Either (`host` and `port`) or `url` must be provided, but not both.'
        if url is not None:
            if host or port:
                raise ValueError(err)
            self.url = url
        else:
            if host is None:
                raise ValueError(err)
            port = port or 80
            self.url = f'http://{host}:{port}'

    def query(self, query: str, k: int=3, metadata_filter: Optional[str]=None) -> List[dict]:
        """
        Perform a query to the vector store and fetch results.

        Args:
            - query:
            - k: number of documents to be returned
            - metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.
        """
        data = {'query': query, 'k': k}
        if metadata_filter is not None:
            data['metadata_filter'] = metadata_filter
        url = self.url + '/v1/retrieve'
        response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'}, timeout=3)
        responses = response.json()
        return sorted(responses, key=lambda x: x['dist'])
    __call__ = query

    def get_vectorstore_statistics(self) -> dict:
        """Fetch basic statistics about the vector store."""
        url = self.url + '/v1/statistics'
        response = requests.post(url, json={}, headers={'Content-Type': 'application/json'})
        responses = response.json()
        return responses

    def get_input_files(self, metadata_filter: Optional[str]=None, filepath_globpattern: Optional[str]=None) -> list:
        """
        Fetch information on documents in the the vector store.

        Args:
            metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.
            filepath_globpattern: optional glob pattern specifying which documents
                will be searched for this query.
        """
        url = self.url + '/v1/inputs'
        response = requests.post(url, json={'metadata_filter': metadata_filter, 'filepath_globpattern': filepath_globpattern}, headers={'Content-Type': 'application/json'})
        responses = response.json()
        return responses