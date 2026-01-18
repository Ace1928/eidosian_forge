from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _UpdateEnv(manifest, remove_container_env, container_env_file, container_env):
    """Update environment variables in container manifest."""
    current_env = {}
    for env_val in manifest['spec']['containers'][0].get('env', []):
        current_env[env_val['name']] = env_val.get('value')
    for env in remove_container_env:
        current_env.pop(env, None)
    current_env.update(_ReadDictionary(container_env_file))
    for env_var_dict in container_env:
        for env, val in six.iteritems(env_var_dict):
            current_env[env] = val
    if current_env:
        manifest['spec']['containers'][0]['env'] = [{'name': env, 'value': val} for env, val in six.iteritems(current_env)]