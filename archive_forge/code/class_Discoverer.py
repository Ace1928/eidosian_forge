import json
import os
from collections import defaultdict
import hashlib
import tempfile
from functools import partial
import kubernetes.dynamic
import kubernetes.dynamic.discovery
from kubernetes import __version__
from kubernetes.dynamic.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.client.resource import (
class Discoverer(kubernetes.dynamic.discovery.Discoverer):

    def __init__(self, client, cache_file):
        self.client = client
        default_cache_file_name = 'k8srcp-{0}.json'.format(hashlib.sha256(self.__get_default_cache_id()).hexdigest())
        self.__cache_file = cache_file or os.path.join(tempfile.gettempdir(), default_cache_file_name)
        self.__init_cache()

    def __get_default_cache_id(self):
        user = self.__get_user()
        if user:
            cache_id = '{0}-{1}'.format(self.client.configuration.host, user)
        else:
            cache_id = self.client.configuration.host
        return cache_id.encode('utf-8')

    def __get_user(self):
        if hasattr(os, 'getlogin'):
            try:
                user = os.getlogin()
                if user:
                    return str(user)
            except OSError:
                pass
        if hasattr(os, 'getuid'):
            try:
                user = os.getuid()
                if user:
                    return str(user)
            except OSError:
                pass
        user = os.environ.get('USERNAME')
        if user:
            return str(user)
        return None

    def __init_cache(self, refresh=False):
        if refresh or not os.path.exists(self.__cache_file):
            self._cache = {'library_version': __version__}
            refresh = True
        else:
            try:
                with open(self.__cache_file, 'r') as f:
                    self._cache = json.load(f, cls=partial(CacheDecoder, self.client))
                if self._cache.get('library_version') != __version__:
                    self.invalidate_cache()
            except Exception:
                self.invalidate_cache()
        self._load_server_info()
        self.discover()
        if refresh:
            self._write_cache()

    def get_resources_for_api_version(self, prefix, group, version, preferred):
        """returns a dictionary of resources associated with provided (prefix, group, version)"""
        resources = defaultdict(list)
        subresources = defaultdict(dict)
        path = '/'.join(filter(None, [prefix, group, version]))
        try:
            resources_response = self.client.request('GET', path).resources or []
        except ServiceUnavailableError:
            resources_response = []
        resources_raw = list(filter(lambda resource: '/' not in resource['name'], resources_response))
        subresources_raw = list(filter(lambda resource: '/' in resource['name'], resources_response))
        for subresource in subresources_raw:
            resource, name = subresource['name'].split('/', 1)
            subresources[resource][name] = subresource
        for resource in resources_raw:
            for key in ('prefix', 'group', 'api_version', 'client', 'preferred'):
                resource.pop(key, None)
            resourceobj = kubernetes.dynamic.Resource(prefix=prefix, group=group, api_version=version, client=self.client, preferred=preferred, subresources=subresources.get(resource['name']), **resource)
            resources[resource['kind']].append(resourceobj)
            resource_lookup = {'prefix': prefix, 'group': group, 'api_version': version, 'kind': resourceobj.kind, 'name': resourceobj.name}
            resource_list = ResourceList(self.client, group=group, api_version=version, base_kind=resource['kind'], base_resource_lookup=resource_lookup)
            resources[resource_list.kind].append(resource_list)
        return resources

    def get(self, **kwargs):
        """
        Same as search, but will throw an error if there are multiple or no
        results. If there are multiple results and only one is an exact match
        on api_version, that resource will be returned.
        """
        results = self.search(**kwargs)
        if len(results) > 1 and kwargs.get('api_version'):
            results = [result for result in results if result.group_version == kwargs['api_version']]
        if len(results) > 1 and (not all((isinstance(x, ResourceList) for x in results))):
            results = [result for result in results if not isinstance(result, ResourceList)]
        if len(results) > 1 and len(set(((x.group_version, x.kind) for x in results))) == 1:
            if len(set((len(x.verbs) for x in results))) != 1:
                results = [max(results, key=lambda x: len(x.verbs))]
        if len(results) == 1:
            return results[0]
        elif not results:
            raise ResourceNotFoundError('No matches found for {0}'.format(kwargs))
        else:
            raise ResourceNotUniqueError('Multiple matches found for {0}: {1}'.format(kwargs, results))