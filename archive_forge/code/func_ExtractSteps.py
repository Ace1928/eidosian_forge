from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def ExtractSteps(job):
    """Extract the cleaned up step dictionary for all the steps in the job.

  Args:
    job: A Job message.
  Returns:
    A list of cleaned up step dictionaries.
  """
    return [_ExtractStep(step) for step in job.steps]