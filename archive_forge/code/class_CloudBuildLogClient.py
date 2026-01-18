from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
class CloudBuildLogClient(object):
    """Client for interacting with the Cloud Build API (and Cloud Build logs)."""

    def __init__(self):
        self.v2_client = v2_client_util.GetClientInstance()

    def _GetLogFilter(self, region, run_id, run_type, has_worker_pool, create_time):
        if has_worker_pool:
            return self._GetWorkerPoolLogFilter(create_time, run_id, run_type, region)
        else:
            return self._GetNonWorkerPoolLogFilter(create_time, run_id, region)

    def _GetNonWorkerPoolLogFilter(self, create_time, run_id, region):
        return 'timestamp>="{timestamp}" AND labels.location="{region}" AND labels.run_name={run_id}'.format(timestamp=create_time, region=region, run_id=run_id)

    def _GetWorkerPoolLogFilter(self, create_time, run_id, run_type, region):
        run_label = 'taskRun' if run_type == 'taskrun' else 'pipelineRun'
        return '(labels."k8s-pod/tekton.dev/{run_label}"="{run_id}" OR labels."k8s-pod/tekton_dev/{run_label}"="{run_id}") AND timestamp>="{timestamp}" AND resource.labels.location="{region}"'.format(run_label=run_label, run_id=run_id, timestamp=create_time, region=region)

    def ShouldStopTailer(self, log_tailer, run, project, region, run_id, run_type):
        """Checks whether a log tailer should be stopped."""
        while run.completionTime is None:
            run = v2_client_util.GetRun(project, region, run_id, run_type)
            time.sleep(1)
        if log_tailer:
            time.sleep(60)
            log_tailer.Stop()
        return run

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

    def PrintLog(self, project, region, run_id, run_type):
        """Print the logs for a run."""
        run = v2_client_util.GetRun(project, region, run_id, run_type)
        has_worker_pool = bool(run.workerPool)
        log_filter = self._GetLogFilter(region, run_id, run_type, has_worker_pool, run.createTime)
        log_tailer = GCLLogTailer.FromFilter(project, region, log_filter, has_worker_pool)
        if log_tailer:
            log_tailer.Print()