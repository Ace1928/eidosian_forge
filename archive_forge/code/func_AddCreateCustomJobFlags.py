from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags as shared_flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.ai.custom_jobs import custom_jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddCreateCustomJobFlags(parser):
    """Adds flags related to create a custom job."""
    shared_flags.AddRegionResourceArg(parser, 'to create a custom job', prompt_func=region_util.GetPromptForRegionFunc(constants.SUPPORTED_TRAINING_REGIONS))
    shared_flags.TRAINING_SERVICE_ACCOUNT.AddToParser(parser)
    shared_flags.NETWORK.AddToParser(parser)
    shared_flags.ENABLE_WEB_ACCESS.AddToParser(parser)
    shared_flags.ENABLE_DASHBOARD_ACCESS.AddToParser(parser)
    shared_flags.AddKmsKeyResourceArg(parser, 'custom job')
    labels_util.AddCreateLabelsFlags(parser)
    _DISPLAY_NAME.AddToParser(parser)
    _PYTHON_PACKAGE_URIS.AddToParser(parser)
    _CUSTOM_JOB_ARGS.AddToParser(parser)
    _CUSTOM_JOB_COMMAND.AddToParser(parser)
    _PERSISTENT_RESOURCE_ID.AddToParser(parser)
    worker_pool_spec_group = base.ArgumentGroup(help='Worker pool specification.', required=True)
    worker_pool_spec_group.AddArgument(_CUSTOM_JOB_CONFIG)
    worker_pool_spec_group.AddArgument(_WORKER_POOL_SPEC)
    worker_pool_spec_group.AddToParser(parser)