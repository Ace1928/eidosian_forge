from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def BuildTargetConfigFromManifest(client, messages, project_id, deployment_id, manifest_id, properties=None):
    """Construct a TargetConfig from a manifest of a previous deployment.

  Args:
    client: Deployment Manager v2 API client.
    messages: Object with v2 API messages.
    project_id: Project for this deployment. This is used when pulling the
        the existing manifest.
    deployment_id: Deployment used to pull retrieve the manifest.
    manifest_id: The manifest to pull down for constructing the target.
    properties: Dictionary of properties, only used if the manifest has a
        single resource. Properties will override only. If the manifest
        has properties which do not exist in the properties hash will remain
        unchanged.

  Returns:
    TargetConfig containing the contents of the config file and the names and
    contents of any imports.

  Raises:
    HttpException: in the event that there is a failure to pull the manifest
        from deployment manager
    ManifestError: When the manifest being revived has more than one
        resource
  """
    try:
        manifest = client.manifests.Get(messages.DeploymentmanagerManifestsGetRequest(project=project_id, deployment=deployment_id, manifest=manifest_id))
        config_file = manifest.config
        imports = manifest.imports
    except apitools_exceptions.HttpError as error:
        raise api_exceptions.HttpException(error)
    if properties:
        config_yaml = yaml.load(config_file.content)
        if len(config_yaml['resources']) != 1:
            raise exceptions.ManifestError('Manifest reuse with properties requires there only be a single resource.')
        single_resource = config_yaml['resources'][0]
        if 'properties' not in single_resource:
            single_resource['properties'] = {}
        existing_properties = single_resource['properties']
        for key, value in six.iteritems(properties):
            existing_properties[key] = value
        config_file.content = yaml.dump(config_yaml)
    return messages.TargetConfiguration(config=config_file, imports=imports)