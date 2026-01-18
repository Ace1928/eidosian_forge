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
def AddTrafficTagsFlags(parser):
    """Add flags for updating traffic tags for a service."""
    AddMapFlagsNoFile(parser, group_help='Specify traffic tags. Traffic tags can be assigned to a revision by name or to the latest ready revision. Assigning a tag to a revision generates a URL prefixed with the tag that allows addressing that revision directly, regardless of the percent traffic specified. Keys are tags. Values are revision names or "LATEST" for the latest ready revision. For example, --set-tags=candidate=LATEST,current=myservice-v1 assigns the tag "candidate" to the latest ready revision and the tag "current" to the revision with name "myservice-v1" and clears any existing tags. Changing tags does not affect the traffic percentage assigned to revisions. When using a tags flag and one or more of --to-latest and --to-revisions in the same command, the tags change occurs first then the traffic percentage change occurs.', flag_name='tags', key_metavar='TAG', value_metavar='REVISION')