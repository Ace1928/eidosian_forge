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
def AddUpdateInstanceSplitFlags(parser):
    """Add flags for updating instance assignments for a worker."""

    @staticmethod
    def InstanceSplitTargetKey(key):
        return key

    @staticmethod
    def InstanceSplitPercentageValue(value):
        """Type validation for intance split percentage flag values."""
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise serverless_exceptions.ArgumentError('Instance split percentage value %s is not an integer.' % value)
        if result < 0 or result > 100:
            raise serverless_exceptions.ArgumentError('Instance split percentage value %s is not between 0 and 100.' % value)
        return result
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--to-revisions', metavar='REVISION-NAME=PERCENTAGE', action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(key_type=InstanceSplitTargetKey.__func__, value_type=InstanceSplitPercentageValue.__func__), help='Comma separated list of instance assignments in the form REVISION-NAME=PERCENTAGE. REVISION-NAME must be the name for a revision for the worker as returned by \'gcloud run workers revisions list --worker=WORKER\' . PERCENTAGE must be an integer percentage between 0 and 100 inclusive.  Ex worker-nw9hs=10,worker-nw9hs=20 Up to 100 percent of instances may be assigned. If the total of 100 percent of instances is assigned, the Worker instance split is updated as specified. If under 100 percent of instance split is assigned, the Worker instance split is updated as specified for revisions with assignments and instance split is scaled up or down proportionally as needed for revision that are currently serving workload but that do not have new assignments. For example assume revision-1 is serving 40 percent of workload and revision-2 is serving 60 percent. If revision-1 is assigned 45 percent of instances and no assignment is made for revision-2, the worker is updated with revsion-1 assigned 45 percent of instances and revision-2 scaled down to 55 percent. You can use "LATEST" as a special revision name to always put the given percentage of instance split on the latest ready revision.')
    group.add_argument('--to-latest', default=False, action='store_true', help="True to assign 100 percent of instances to the 'latest' revision of this service. Note that when a new revision is created, it will become the 'latest' and instances will be fully assigned to it unless configured otherwise using `--[no-]promote` flag. Defaults to False. Synonymous with '--to-revisions=LATEST=100'.")