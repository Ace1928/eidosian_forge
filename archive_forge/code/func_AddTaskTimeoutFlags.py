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
def AddTaskTimeoutFlags(parser, for_execution_overrides=False):
    """Add job flags for job and task deadline."""
    help_text = ' In the case of retries, this deadline applies to each attempt of a task. If the task attempt does not complete within this time, it will be killed. It is specified as a duration; for example, "10m5s" is ten minutes, and five seconds. If you don\'t specify a unit, seconds is assumed. For example, "10" is 10 seconds.'
    if for_execution_overrides:
        help_text = 'The existing maximum time (deadline) a job task attempt can run for. If provided, an execution will be created with this value. Otherwise existing maximum time of the job is used.' + help_text
    else:
        help_text = 'Set the maximum time (deadline) a job task attempt can run for.' + help_text
    parser.add_argument('--task-timeout', type=arg_parsers.Duration(lower_bound='1s'), help=help_text)