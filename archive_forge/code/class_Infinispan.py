from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
class Infinispan:
    """Helper class for `Infinispan` REST interface.

    This class exposes the Infinispan operations needed to
    create and set up a vector db.

    You need a running Infinispan (15+) server without authentication.
    You can easily start one, see:
    https://github.com/rigazilla/infinispan-vector#run-infinispan
    """

    def __init__(self, **kwargs: Any):
        self._configuration = kwargs
        self._schema = str(self._configuration.get('schema', 'http'))
        self._host = str(self._configuration.get('hosts', ['127.0.0.1:11222'])[0])
        self._default_node = self._schema + '://' + self._host
        self._cache_url = str(self._configuration.get('cache_url', '/rest/v2/caches'))
        self._schema_url = str(self._configuration.get('cache_url', '/rest/v2/schemas'))
        self._use_post_for_query = str(self._configuration.get('use_post_for_query', True))

    def req_query(self, query: str, cache_name: str, local: bool=False) -> requests.Response:
        """Request a query
        Args:
            query(str): query requested
            cache_name(str): name of the target cache
            local(boolean): whether the query is local to clustered
        Returns:
            An http Response containing the result set or errors
        """
        if self._use_post_for_query:
            return self._query_post(query, cache_name, local)
        return self._query_get(query, cache_name, local)

    def _query_post(self, query_str: str, cache_name: str, local: bool=False) -> requests.Response:
        api_url = self._default_node + self._cache_url + '/' + cache_name + '?action=search&local=' + str(local)
        data = {'query': query_str}
        data_json = json.dumps(data)
        response = requests.post(api_url, data_json, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
        return response

    def _query_get(self, query_str: str, cache_name: str, local: bool=False) -> requests.Response:
        api_url = self._default_node + self._cache_url + '/' + cache_name + '?action=search&query=' + query_str + '&local=' + str(local)
        response = requests.get(api_url, timeout=REST_TIMEOUT)
        return response

    def post(self, key: str, data: str, cache_name: str) -> requests.Response:
        """Post an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '/' + key
        response = requests.post(api_url, data, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
        return response

    def put(self, key: str, data: str, cache_name: str) -> requests.Response:
        """Put an entry
        Args:
            key(str): key of the entry
            data(str): content of the entry in json format
            cache_name(str): target cache
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '/' + key
        response = requests.put(api_url, data, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
        return response

    def get(self, key: str, cache_name: str) -> requests.Response:
        """Get an entry
        Args:
            key(str): key of the entry
            cache_name(str): target cache
        Returns:
            An http Response containing the entry or errors
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '/' + key
        response = requests.get(api_url, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
        return response

    def schema_post(self, name: str, proto: str) -> requests.Response:
        """Deploy a schema
        Args:
            name(str): name of the schema. Will be used as a key
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + '/' + name
        response = requests.post(api_url, proto, timeout=REST_TIMEOUT)
        return response

    def cache_post(self, name: str, config: str) -> requests.Response:
        """Create a cache
        Args:
            name(str): name of the cache.
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + name
        response = requests.post(api_url, config, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
        return response

    def schema_delete(self, name: str) -> requests.Response:
        """Delete a schema
        Args:
            name(str): name of the schema.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._schema_url + '/' + name
        response = requests.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_delete(self, name: str) -> requests.Response:
        """Delete a cache
        Args:
            name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + name
        response = requests.delete(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_clear(self, cache_name: str) -> requests.Response:
        """Clear a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '?action=clear'
        response = requests.post(api_url, timeout=REST_TIMEOUT)
        return response

    def cache_exists(self, cache_name: str) -> bool:
        """Check if a cache exists
        Args:
            cache_name(str): name of the cache.
        Returns:
            True if cache exists
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '?action=clear'
        return self.resource_exists(api_url)

    @staticmethod
    def resource_exists(api_url: str) -> bool:
        """Check if a resource exists
        Args:
            api_url(str): url of the resource.
        Returns:
            true if resource exists
        """
        response = requests.head(api_url, timeout=REST_TIMEOUT)
        return response.ok

    def index_clear(self, cache_name: str) -> requests.Response:
        """Clear an index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '/search/indexes?action=clear'
        return requests.post(api_url, timeout=REST_TIMEOUT)

    def index_reindex(self, cache_name: str) -> requests.Response:
        """Rebuild index on a cache
        Args:
            cache_name(str): name of the cache.
        Returns:
            An http Response containing the result of the operation
        """
        api_url = self._default_node + self._cache_url + '/' + cache_name + '/search/indexes?action=reindex'
        return requests.post(api_url, timeout=REST_TIMEOUT)