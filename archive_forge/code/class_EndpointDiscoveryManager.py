import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
class EndpointDiscoveryManager:

    def __init__(self, client, cache=None, current_time=None, always_discover=True):
        if cache is None:
            cache = {}
        self._cache = cache
        self._failed_attempts = {}
        if current_time is None:
            current_time = time.time
        self._time = current_time
        self._always_discover = always_discover
        self._client = weakref.proxy(client)
        self._model = EndpointDiscoveryModel(client.meta.service_model)

    def _parse_endpoints(self, response):
        endpoints = response['Endpoints']
        current_time = self._time()
        for endpoint in endpoints:
            cache_time = endpoint.get('CachePeriodInMinutes')
            endpoint['Expiration'] = current_time + cache_time * 60
        return endpoints

    def _cache_item(self, value):
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        else:
            return value

    def _create_cache_key(self, **kwargs):
        kwargs = self._model.discovery_operation_kwargs(**kwargs)
        return tuple((self._cache_item(v) for k, v in sorted(kwargs.items())))

    def gather_identifiers(self, operation, params):
        return self._model.gather_identifiers(operation, params)

    def delete_endpoints(self, **kwargs):
        cache_key = self._create_cache_key(**kwargs)
        if cache_key in self._cache:
            del self._cache[cache_key]

    def _describe_endpoints(self, **kwargs):
        kwargs = self._model.discovery_operation_kwargs(**kwargs)
        operation_name = self._model.discovery_operation_name
        discovery_operation = getattr(self._client, operation_name)
        logger.debug('Discovering endpoints with kwargs: %s', kwargs)
        return discovery_operation(**kwargs)

    def _get_current_endpoints(self, key):
        if key not in self._cache:
            return None
        now = self._time()
        return [e for e in self._cache[key] if now < e['Expiration']]

    def _refresh_current_endpoints(self, **kwargs):
        cache_key = self._create_cache_key(**kwargs)
        try:
            response = self._describe_endpoints(**kwargs)
            endpoints = self._parse_endpoints(response)
            self._cache[cache_key] = endpoints
            self._failed_attempts.pop(cache_key, None)
            return endpoints
        except (ConnectionError, HTTPClientError):
            self._failed_attempts[cache_key] = self._time() + 60
            return None

    def _recently_failed(self, cache_key):
        if cache_key in self._failed_attempts:
            now = self._time()
            if now < self._failed_attempts[cache_key]:
                return True
            del self._failed_attempts[cache_key]
        return False

    def _select_endpoint(self, endpoints):
        return endpoints[0]['Address']

    def describe_endpoint(self, **kwargs):
        operation = kwargs['Operation']
        discovery_required = self._model.discovery_required_for(operation)
        if not self._always_discover and (not discovery_required):
            logger.debug('Optional discovery disabled. Skipping discovery for Operation: %s' % operation)
            return None
        cache_key = self._create_cache_key(**kwargs)
        endpoints = self._get_current_endpoints(cache_key)
        if endpoints:
            return self._select_endpoint(endpoints)
        recently_failed = self._recently_failed(cache_key)
        if not recently_failed:
            endpoints = self._refresh_current_endpoints(**kwargs)
            if endpoints:
                return self._select_endpoint(endpoints)
        logger.debug('Endpoint Discovery has failed for: %s', kwargs)
        stale_entries = self._cache.get(cache_key, None)
        if stale_entries:
            return self._select_endpoint(stale_entries)
        if discovery_required:
            if recently_failed:
                endpoints = self._refresh_current_endpoints(**kwargs)
                if endpoints:
                    return self._select_endpoint(endpoints)
            raise EndpointDiscoveryRefreshFailed()
        return None