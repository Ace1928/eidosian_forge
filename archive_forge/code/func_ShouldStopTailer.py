from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
def ShouldStopTailer(self, log_tailer, run, project, region, run_id, run_type):
    """Checks whether a log tailer should be stopped."""
    while run.completionTime is None:
        run = v2_client_util.GetRun(project, region, run_id, run_type)
        time.sleep(1)
    if log_tailer:
        time.sleep(60)
        log_tailer.Stop()
    return run