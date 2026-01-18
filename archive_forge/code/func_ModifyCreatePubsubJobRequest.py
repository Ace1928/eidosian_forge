from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def ModifyCreatePubsubJobRequest(job_ref, args, create_job_req):
    """Add the pubsubMessage field to the given request.

  Because the Cloud Scheduler API has a reference to a PubSub message, but
  represents it as a bag of properties, we need to construct the object here and
  insert it into the request.

  Args:
    job_ref: Resource reference to the job to be created (unused)
    args: argparse namespace with the parsed arguments from the command line. In
        particular, we expect args.message_body and args.attributes (optional)
        to be AdditionalProperty types.
    create_job_req: CloudschedulerProjectsLocationsJobsCreateRequest, the
        request constructed from the remaining arguments.

  Returns:
    CloudschedulerProjectsLocationsJobsCreateRequest: the given request but with
        the job.pubsubTarget.pubsubMessage field populated.
  """
    ModifyCreateJobRequest(job_ref, args, create_job_req)
    create_job_req.job.pubsubTarget.data = _EncodeMessageBody(args.message_body or args.message_body_from_file)
    if args.attributes:
        create_job_req.job.pubsubTarget.attributes = args.attributes
    return create_job_req