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
def AddTaskFilterFlags(parser):
    """Add filter flags for task list."""
    parser.add_argument('--succeeded', action='append_const', dest='filter_flags', const='Succeeded', help='Include succeeded tasks.')
    parser.add_argument('--failed', action='append_const', dest='filter_flags', const='Failed', help='Include failed tasks.')
    parser.add_argument('--cancelled', action='append_const', dest='filter_flags', const='Cancelled', help='Include cancelled tasks.')
    parser.add_argument('--running', action='append_const', dest='filter_flags', const='Running', help='Include running tasks.')
    parser.add_argument('--abandoned', action='append_const', dest='filter_flags', const='Abandoned', help='Include abandoned tasks.')
    parser.add_argument('--pending', action='append_const', dest='filter_flags', const='Pending', help='Include pending tasks.')
    parser.add_argument('--completed', action=arg_parsers.ExtendConstAction, dest='filter_flags', const=['Succeeded', 'Failed', 'Cancelled'], help='Include succeeded, failed, and cancelled tasks.')
    parser.add_argument('--no-completed', action=arg_parsers.ExtendConstAction, dest='filter_flags', const=['Running', 'Pending'], help='Include running and pending tasks.')
    parser.add_argument('--started', action=arg_parsers.ExtendConstAction, dest='filter_flags', const=['Succeeded', 'Failed', 'Cancelled', 'Running'], help='Include running, succeeded, failed, and cancelled tasks.')
    parser.add_argument('--no-started', action=arg_parsers.ExtendConstAction, dest='filter_flags', const=['Pending', 'Abandoned'], help='Include pending and abandoned tasks.')