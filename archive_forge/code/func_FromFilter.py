from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
@classmethod
def FromFilter(cls, project, location, log_filter, has_worker_pool, out=log.out):
    """Build a GCLLogTailer from a log filter."""
    return cls(project=project, log_filter=log_filter, location=location, has_worker_pool=has_worker_pool, out=out)