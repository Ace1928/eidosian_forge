from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import OrderedDict
import json
import re
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core.util import files
import six
class DiscoveryDoc(object):
    """Encapsulates access to discovery doc."""

    def __init__(self, discovery_doc_dict):
        self._discovery_doc_dict = discovery_doc_dict

    @classmethod
    def FromJson(cls, path):
        with files.FileReader(path) as f:
            return cls(json.load(f, object_pairs_hook=OrderedDict))

    @property
    def api_name(self):
        return self._discovery_doc_dict['name']

    @property
    def api_version(self):
        return self._discovery_doc_dict['version']

    @property
    def base_url(self):
        return self._discovery_doc_dict['baseUrl']

    @property
    def docs_url(self):
        return self._discovery_doc_dict['documentationLink']

    def GetResourceCollections(self, custom_resources, api_version):
        """Returns all resources collections found in this discovery doc.

    Args:
      custom_resources: {str, str}, A mapping of collection name to path that
          have been registered manually in the yaml file.
      api_version: Override api_version for each found resource collection.

    Returns:
      list(resource_util.CollectionInfo).
    """
        collections = self._ExtractResources(api_version, self._discovery_doc_dict)
        collections.extend(self._GenerateMissingParentCollections(collections, custom_resources, api_version))
        return collections

    def _ExtractResources(self, api_version, infos):
        """Extract resource definitions from discovery doc."""
        collections = []
        if infos.get('methods'):
            methods = infos.get('methods')
            get_method = methods.get('get')
            if get_method:
                collection_info = self._GetCollectionFromMethod(api_version, get_method)
                collections.append(collection_info)
        if infos.get('resources'):
            for _, info in infos.get('resources').items():
                subresource_collections = self._ExtractResources(api_version, info)
                collections.extend(subresource_collections)
        return collections

    def _GetCollectionFromMethod(self, api_version, get_method):
        """Created collection_info object given discovery doc get_method."""
        collection_name = _ExtractCollectionName(get_method['id'])
        collection_name = collection_name.split('.', 1)[1]
        flat_path = get_method.get('flatPath')
        path = get_method.get('path')
        return self._MakeResourceCollection(api_version, collection_name, path, flat_path)

    def _MakeResourceCollection(self, api_version, collection_name, path, flat_path=None):
        """Make resource collection object given its name and path."""
        if flat_path == path:
            flat_path = None
        url = self.base_url + path
        url_api_name, url_api_version, path = resource_util.SplitEndpointUrl(url)
        if url_api_version != api_version:
            raise UnsupportedDiscoveryDoc('Collection {0} for version {1}/{2} is using url {3} with version {4}'.format(collection_name, self.api_name, api_version, url, url_api_version))
        if flat_path:
            _, _, flat_path = resource_util.SplitEndpointUrl(self.base_url + flat_path)
        url = url[:-len(path)]
        return resource_util.CollectionInfo(url_api_name, api_version, url, self.docs_url, collection_name, path, {DEFAULT_PATH_NAME: flat_path} if flat_path else {}, resource_util.GetParamsFromPath(path))

    def _GenerateMissingParentCollections(self, collections, custom_resources, api_version):
        """Generates parent collections for any existing collection missing one.

    Args:
      collections: [resource.CollectionInfo], The existing collections from the
        discovery doc.
      custom_resources: {str, str}, A mapping of collection name to path that
        have been registered manually in the yaml file.
      api_version: Override api_version for each found resource collection.

    Raises:
      ConflictingCollection: If multiple parent collections have the same name
        but different paths, and a custom resource has not been declared to
        resolve the conflict.

    Returns:
      [resource.CollectionInfo], Additional collections to include in the
      resource module.
    """
        all_names = {c.name: c for c in collections}
        all_paths = {c.GetPath(DEFAULT_PATH_NAME) for c in collections}
        generated = []
        in_progress = list(collections)
        to_process = []
        ignored = {}
        while in_progress:
            for c in in_progress:
                parent_name, parent_path = _GetParentCollection(c)
                if not parent_name:
                    continue
                if parent_path in all_paths:
                    continue
                if parent_name in custom_resources:
                    ignored.setdefault(parent_name, set()).add(parent_path)
                    continue
                if parent_name in all_names:
                    raise ConflictingCollection('In API [{api}/{version}], the parent of collection [{c}] is not registered, but a collection with [{parent_name}] and path [{existing_path}] already exists. Update the api config file to manually add the parent collection with a path of [{parent_path}].'.format(api=c.api_name, version=api_version, c=c.name, parent_name=parent_name, existing_path=all_names[parent_name].GetPath(DEFAULT_PATH_NAME), parent_path=parent_path))
                parent_collection = self.MakeResourceCollection(parent_name, parent_path, True, api_version)
                to_process.append(parent_collection)
                all_names[parent_name] = parent_collection
                all_paths.add(parent_path)
            generated.extend(to_process)
            in_progress = to_process
            to_process = []
        for name, paths in six.iteritems(ignored):
            if len(paths) > 1:
                continue
            path = paths.pop()
            if path == custom_resources[name]['path']:
                print('WARNING: Custom resource [{}] in API [{}/{}] is redundant.'.format(name, self.api_name, api_version))
        return generated

    def MakeResourceCollection(self, collection_name, path, enable_uri_parsing, api_version):
        _, url_api_version, _ = resource_util.SplitEndpointUrl(self.base_url)
        if url_api_version:
            base_url = self.base_url
        else:
            base_url = '{}{}/'.format(self.base_url, api_version)
        return resource_util.CollectionInfo(self.api_name, api_version, base_url, self.docs_url, collection_name, path, {}, resource_util.GetParamsFromPath(path), enable_uri_parsing)