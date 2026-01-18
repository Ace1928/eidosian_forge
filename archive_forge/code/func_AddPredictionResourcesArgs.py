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
def AddPredictionResourcesArgs(parser, version):
    """Add arguments for prediction resources."""
    base.Argument('--min-replica-count', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), help='Minimum number of machine replicas for the deployment resources the model will be\ndeployed on. If specified, the value must be equal to or larger than 1.\n\nIf not specified and the uploaded models use dedicated resources, the default\nvalue is 1.\n').AddToParser(parser)
    base.Argument('--max-replica-count', type=int, help='Maximum number of machine replicas for the deployment resources the model will be\ndeployed on.\n').AddToParser(parser)
    base.Argument('--machine-type', help='The machine resources to be used for each node of this deployment.\nFor available machine types, see\nhttps://cloud.google.com/ai-platform-unified/docs/predictions/machine-types.\n').AddToParser(parser)
    base.Argument('--accelerator', type=arg_parsers.ArgDict(spec={'type': str, 'count': int}, required_keys=['type']), help='Manage the accelerator config for GPU serving. When deploying a model with\nCompute Engine Machine Types, a GPU accelerator may also\nbe selected.\n\n*type*::: The type of the accelerator. Choices are {}.\n\n*count*::: The number of accelerators to attach to each machine running the job.\n This is usually 1. If not specified, the default value is 1.\n\nFor example:\n`--accelerator=type=nvidia-tesla-k80,count=1`'.format(', '.join(["'{}'".format(c) for c in GetAcceleratorTypeMapper(version).choices]))).AddToParser(parser)