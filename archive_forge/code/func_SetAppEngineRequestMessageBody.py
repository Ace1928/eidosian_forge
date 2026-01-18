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
def SetAppEngineRequestMessageBody(unused_job_ref, args, update_job_req):
    """Modify the App Engine update request to populate the message body."""
    if args.clear_message_body:
        update_job_req.job.appEngineHttpTarget.body = None
    elif args.message_body or args.message_body_from_file:
        update_job_req.job.appEngineHttpTarget.body = _EncodeMessageBody(args.message_body or args.message_body_from_file)
    return update_job_req