from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
def _GetLogFilter(self, region, run_id, run_type, has_worker_pool, create_time):
    if has_worker_pool:
        return self._GetWorkerPoolLogFilter(create_time, run_id, run_type, region)
    else:
        return self._GetNonWorkerPoolLogFilter(create_time, run_id, region)