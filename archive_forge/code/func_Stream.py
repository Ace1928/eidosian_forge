from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
def Stream(self, project, region, run_id, run_type, out=log.out):
    """Streams the logs for a run if available."""
    run = v2_client_util.GetRun(project, region, run_id, run_type)
    has_worker_pool = bool(run.workerPool)
    log_filter = self._GetLogFilter(region, run_id, run_type, has_worker_pool, run.createTime)
    log_tailer = GCLLogTailer.FromFilter(project, region, log_filter, has_worker_pool, out=out)
    t = None
    if log_tailer:
        t = v1_logs_util.ThreadInterceptor(target=log_tailer.Tail)
        t.start()
    run = self.ShouldStopTailer(log_tailer, run, project, region, run_id, run_type)
    if t:
        t.join()
        if t.exception is not None:
            raise t.exception
    return run