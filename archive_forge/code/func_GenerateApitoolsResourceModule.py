from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from apitools.gen import gen_client
from googlecloudsdk.api_lib.regen import api_def
from googlecloudsdk.api_lib.regen import resource_generator
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
import six
def GenerateApitoolsResourceModule(base_dir, root_dir, api_name, api_version, discovery_doc_path, custom_resources):
    """Create resource.py file for given api and its discovery doc.

  Args:
      base_dir: str, Path of directory for the project.
      root_dir: str, Path of the resource file location within the project.
      api_name: str, name of the api.
      api_version: str, the version for the api.
      discovery_doc_path: str, file path to discovery doc.
      custom_resources: dict, dictionary of custom resource collections.
  Raises:
    WrongDiscoveryDocError: if discovery doc api name/version does not match.
  """
    discovery_doc = resource_generator.DiscoveryDoc.FromJson(os.path.join(base_dir, root_dir, discovery_doc_path))
    if discovery_doc.api_version != api_version:
        logging.warning('Discovery api version %s does not match %s, this client will be accessible via new alias.', discovery_doc.api_version, api_version)
    if discovery_doc.api_name != api_name:
        raise WrongDiscoveryDocError('api name {0}, expected {1}'.format(discovery_doc.api_name, api_name))
    resource_collections = discovery_doc.GetResourceCollections(custom_resources, api_version)
    if custom_resources:
        matched_resources = set([])
        for collection in resource_collections:
            if collection.name in custom_resources:
                apitools_compatible = custom_resources[collection.name].get('apitools_compatible', True)
                if not apitools_compatible:
                    continue
                matched_resources.add(collection.name)
                custom_path = custom_resources[collection.name]['path']
                if isinstance(custom_path, dict):
                    collection.flat_paths.update(custom_path)
                elif isinstance(custom_path, six.string_types):
                    collection.flat_paths[resource_generator.DEFAULT_PATH_NAME] = custom_path
        for collection_name in set(custom_resources.keys()) - matched_resources:
            collection_def = custom_resources[collection_name]
            collection_path = collection_def['path']
            apitools_compatible = collection_def.get('apitools_compatible', True)
            if not apitools_compatible:
                continue
            enable_uri_parsing = collection_def.get('enable_uri_parsing', True)
            collection_info = discovery_doc.MakeResourceCollection(collection_name, collection_path, enable_uri_parsing, api_version)
            resource_collections.append(collection_info)
    api_dir = os.path.join(base_dir, root_dir, api_name, api_version)
    if not os.path.exists(api_dir):
        os.makedirs(api_dir)
    resource_file_name = os.path.join(api_dir, 'resources.py')
    if resource_collections:
        logging.debug('Generating resource module at %s', resource_file_name)
        tpl = template.Template(filename=os.path.join(os.path.dirname(__file__), 'resources.tpl'))
        with files.FileWriter(resource_file_name) as output_file:
            ctx = runtime.Context(output_file, collections=sorted(resource_collections), base_url=resource_collections[0].base_url, docs_url=discovery_doc.docs_url)
            tpl.render_context(ctx)
    elif os.path.isfile(resource_file_name):
        logging.debug('Removing existing resource module at %s', resource_file_name)
        os.remove(resource_file_name)