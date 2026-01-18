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
def AddConfigMapsFlags(parser):
    """Adds flags for creating, updating, and deleting config maps."""
    AddMapFlagsNoFile(parser, group_help="Specify config map to mount or provide as environment variables. Keys starting with a forward slash '/' are mount paths. All other keys correspond to environment variables. The values associated with each of these should be in the form CONFIG_MAP_NAME:KEY_IN_CONFIG_MAP; you may omit the key within the config map to specify a mount of all keys within the config map. For example: '--update-config-maps=/my/path=myconfig,ENV=otherconfig:key.json' will create a volume with config map 'myconfig' and mount that volume at '/my/path'. Because no config map key was specified, all keys in 'myconfig' will be included. An environment variable named ENV will also be created whose value is the value of 'key.json' in 'otherconfig. Not supported on the fully managed version of Cloud Run.", flag_name='config-maps')