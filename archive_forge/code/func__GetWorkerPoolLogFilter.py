from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.cloudbuild import logs as v1_logs_util
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util as v2_client_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import log
def _GetWorkerPoolLogFilter(self, create_time, run_id, run_type, region):
    run_label = 'taskRun' if run_type == 'taskrun' else 'pipelineRun'
    return '(labels."k8s-pod/tekton.dev/{run_label}"="{run_id}" OR labels."k8s-pod/tekton_dev/{run_label}"="{run_id}") AND timestamp>="{timestamp}" AND resource.labels.location="{region}"'.format(run_label=run_label, run_id=run_id, timestamp=create_time, region=region)