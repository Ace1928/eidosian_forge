from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import datetime
import io
import os.path
import shutil
import tarfile
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import staging_bucket_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _CreateSkaffoldFileForRunContainer(target_to_target_properties):
    """Creates skaffold file for target_ids in _TargetProperties object.

  Args:
    target_to_target_properties: A dict of target_id to _TargetProperties.

  Returns:
    skaffold yaml.

  """
    skaffold = collections.OrderedDict()
    skaffold['apiVersion'] = 'skaffold/v3alpha1'
    skaffold['kind'] = 'Config'
    if len(target_to_target_properties) == 1:
        skaffold['manifests'] = {'rawYaml': ['{}_manifest.yaml'.format(target_to_target_properties.keys()[0])]}
    else:
        skaffold['profiles'] = []
        for target_id in target_to_target_properties:
            skaffold['profiles'].append(collections.OrderedDict([('name', target_to_target_properties[target_id].profile), ('manifests', {'rawYaml': ['{}_manifest.yaml'.format(target_id)]})]))
    skaffold['deploy'] = {'cloudrun': {}}
    return skaffold