from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetDisableContainerLoggingArg():
    return base.Argument('--disable-container-logging', action='store_true', default=False, required=False, help='For custom-trained Models and AutoML Tabular Models, the container of the\ndeployed model instances will send `stderr` and `stdout` streams to\nCloud Logging by default. Please note that the logs incur cost,\nwhich are subject to [Cloud Logging\npricing](https://cloud.google.com/stackdriver/pricing).\n\nUser can disable container logging by setting this flag to true.\n')