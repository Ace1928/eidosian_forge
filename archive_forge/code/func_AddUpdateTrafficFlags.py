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
def AddUpdateTrafficFlags(parser):
    """Add flags for updating traffic assignments for a service."""

    @staticmethod
    def TrafficTargetKey(key):
        return key

    @staticmethod
    def TrafficPercentageValue(value):
        """Type validation for traffic percentage flag values."""
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise serverless_exceptions.ArgumentError('Traffic percentage value %s is not an integer.' % value)
        if result < 0 or result > 100:
            raise serverless_exceptions.ArgumentError('Traffic percentage value %s is not between 0 and 100.' % value)
        return result
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--to-revisions', metavar='REVISION-NAME=PERCENTAGE', action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(key_type=TrafficTargetKey.__func__, value_type=TrafficPercentageValue.__func__), help='Comma separated list of traffic assignments in the form REVISION-NAME=PERCENTAGE. REVISION-NAME must be the name for a revision for the service as returned by \'gcloud beta run list revisions\'. PERCENTAGE must be an integer percentage between 0 and 100 inclusive.  Ex service-nw9hs=10,service-nw9hs=20 Up to 100 percent of traffic may be assigned. If 100 percent of traffic is assigned,  the Service traffic is updated as specified. If under 100 percent of traffic is assigned, the Service traffic is updated as specified for revisions with assignments and traffic is scaled up or down down proportionally as needed for revision that are currently serving traffic but that do not have new assignments. For example assume revision-1 is serving 40 percent of traffic and revision-2 is serving 60 percent. If revision-1 is assigned 45 percent of traffic and no assignment is made for revision-2, the service is updated with revsion-1 assigned 45 percent of traffic and revision-2 scaled down to 55 percent. You can use "LATEST" as a special revision name to always put the given percentage of traffic on the latest ready revision.')
    group.add_argument('--to-tags', metavar='TAG=PERCENTAGE', action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(key_type=TrafficTargetKey.__func__, value_type=TrafficPercentageValue.__func__), help='Comma separated list of traffic assignments in the form TAG=PERCENTAGE. TAG must match a traffic tag on a revision of the service. It may match a previously-set tag, or one assigned using the `--set-tags` or `--update-tags` flags on this command. PERCENTAGE must be an integer percentage between 0 and 100 inclusive. Up to 100 percent of traffic may be assigned. If 100 percent of traffic is assigned, the service traffic is updated as specified. If under 100 percent of traffic is assigned, the service traffic is updated as specified to the given tags, and other traffic is scaled up or down proportionally. For example, assume the revision tagged `next` is serving 40 percent of traffic and the revision tagged `current` is serving 60 percent. If `next` is assigned 45 percent of traffic and no assignment is made for `current`, the service is updated with `next` assigned 45 percent of traffic and `current` scaled down to 55 percent. ')
    group.add_argument('--to-latest', default=False, action='store_true', help="True to assign 100 percent of traffic to the 'latest' revision of this service. Note that when a new revision is created, it will become the 'latest' and traffic will be directed to it. Defaults to False. Synonymous with '--to-revisions=LATEST=100'.")