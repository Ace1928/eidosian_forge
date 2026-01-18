from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from googlecloudsdk.api_lib.batch import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.batch import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _GetJobId(self, job_ref, args):
    job_id = job_ref.RelativeName().split('/')[-1]
    if job_id != resource_args.INVALIDID and args.job_prefix:
        raise exceptions.Error('--job-prefix cannot be specified when JOB ID positional argument is specified')
    elif args.job_prefix:
        job_id = args.job_prefix + '-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    elif job_id == resource_args.INVALIDID:
        job_id = None
    return job_id