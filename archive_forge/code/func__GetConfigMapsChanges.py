import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _GetConfigMapsChanges(args):
    """Return config map env var and volume changes for given args."""
    volume_kwargs = {}
    env_kwargs = {}
    updates = _StripKeys(getattr(args, 'update_config_maps', None) or args.set_config_maps or {})
    volume_kwargs['updates'] = {k: v for k, v in updates.items() if _IsVolumeMountKey(k)}
    env_kwargs['updates'] = {k: v for k, v in updates.items() if not _IsVolumeMountKey(k)}
    removes = _MapLStrip(getattr(args, 'remove_config_maps', None) or [])
    volume_kwargs['removes'] = [k for k in removes if _IsVolumeMountKey(k)]
    env_kwargs['removes'] = [k for k in removes if not _IsVolumeMountKey(k)]
    clear_others = bool(args.set_config_maps or args.clear_config_maps)
    env_kwargs['clear_others'] = clear_others
    volume_kwargs['clear_others'] = clear_others
    config_maps_changes = []
    if any(env_kwargs.values()):
        config_maps_changes.append(config_changes.ConfigMapEnvVarChanges(**env_kwargs))
    if any(volume_kwargs.values()):
        config_maps_changes.append(config_changes.ConfigMapVolumeChanges(**volume_kwargs))
    return config_maps_changes