from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
def _GetExecutionUiLink(execution):
    return 'https://console.cloud.google.com/run/jobs/executions/details/{region}/{execution_name}/tasks?project={project}'.format(region=execution.region, execution_name=execution.name, project=execution.namespace)