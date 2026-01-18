from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.dataproc import clusters
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
@staticmethod
def ValidateArgs(args):
    super(CreateBeta, CreateBeta).ValidateArgs(args)
    if args.master_accelerator and 'type' not in args.master_accelerator:
        raise exceptions.InvalidArgumentException('--master-accelerator', 'accelerator type must be specified. e.g. --master-accelerator type=nvidia-tesla-k80,count=2')
    if args.worker_accelerator and 'type' not in args.worker_accelerator:
        raise exceptions.InvalidArgumentException('--worker-accelerator', 'accelerator type must be specified. e.g. --worker-accelerator type=nvidia-tesla-k80,count=2')