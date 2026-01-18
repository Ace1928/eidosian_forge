from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def ParseName(pattern, primitive_type):
    """Parses the name of a pipelineRun/taskRun.

  Args:
    pattern:
      "projects/{project}/locations/{location}/pipelineRuns/{pipeline_run}"
      "projects/{project}/locations/{location}/taskRuns/{task_run}"
    primitive_type: string

  Returns:
    name: string
  """
    if primitive_type == 'pipelinerun':
        match = re.match('projects/([^/]+)/locations/([^/]+)/pipelineRuns/([^/]+)', pattern)
        if match:
            return match.group(3)
    elif primitive_type == 'taskrun':
        match = re.match('projects/([^/]+)/locations/([^/]+)/taskRuns/([^/]+)', pattern)
        if match:
            return match.group(3)