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
def AddVolumesFlags(parser, release_track):
    """Add flags for adding and removing volumes."""
    group = parser.add_group()
    group.add_argument('--add-volume', type=arg_parsers.ArgDict(required_keys=['name', 'type']), action='append', metavar='KEY=VALUE', help='Adds a volume to the Cloud Run resource. To add more than one volume, specify this flag multiple times. Volumes must have a `name` and `type` key. Only certain values are supported for `type`. Depending on the provided type, other keys will be required. The following types are supported with the specified additional keys:\n\n' + volumes.volume_help(release_track))
    group.add_argument('--remove-volume', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, metavar='VOLUME', help='Removes volumes from the Cloud Run resource.')
    group.add_argument('--clear-volumes', action='store_true', help='Remove all existing volumes from the Cloud Run resource, including volumes mounted as secrets')