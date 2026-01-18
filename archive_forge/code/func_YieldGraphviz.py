from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def YieldGraphviz(steps, graph_name=None):
    """Given a root cluster produce the Graphviz DOT format.

  No attempt is made to produce `pretty' output.

  Args:
    steps: A list of steps from the Job message.
    graph_name: The name of the graph to output.

  Yields:
    The lines representing the step-graph in Graphviz format.
  """
    yield 'strict digraph {graph_name} {{'.format(graph_name=_EscapeGraphvizId(graph_name or 'G'))
    root = _UnflattenStepsToClusters(steps)
    for line in _YieldGraphvizClusters(root):
        yield line
    yield ''
    for step in steps:
        for line in _YieldGraphvizEdges(step):
            yield line
    yield '}'