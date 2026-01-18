from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def _YieldGraphvizEdges(step):
    """Output Graphviz edges for the given step.

  Args:
    step: Step to write edges for.

  Yields:
    The Graphviz edge lines for the given step.
  """
    step_name = step['name']
    par_input = step['properties'].get('parallel_input', None)
    if par_input:
        yield _GraphvizEdge(step_name, par_input)
    for other_input in step['properties'].get('inputs', []):
        yield _GraphvizEdge(step_name, other_input)
    for side_input in step['properties'].get('non_parallel_inputs', {}).values():
        yield _GraphvizEdge(step_name, side_input, style='dashed')