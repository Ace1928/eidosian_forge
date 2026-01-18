from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _DisplayExecutionStats(self, out, prepend, beneath_stub):
    """Prints the relevant execution statistics for a node.

    More specifically, print out latency information and the number of
    executions. This information only exists when query is run in 'PROFILE'
    mode.

    Args:
      out: Output stream to which we print.
      prepend: String that precedes any information about this node to maintain
        a visible hierarchy.
      beneath_stub: String that preserves the indentation of the vertical lines.
    """
    if not self.properties.executionStats:
        return None
    stat_props = []
    num_executions = self._GetNestedStatProperty('execution_summary', 'num_executions')
    if num_executions:
        num_executions = int(num_executions)
        executions_str = '{} {}'.format(num_executions, text.Pluralize(num_executions, 'execution'))
        stat_props.append(executions_str)
    mean_latency = self._GetNestedStatProperty('latency', 'mean')
    total_latency = self._GetNestedStatProperty('latency', 'total')
    unit = self._GetNestedStatProperty('latency', 'unit')
    if mean_latency:
        stat_props.append('{} {} average latency'.format(mean_latency, unit))
    elif total_latency:
        stat_props.append('{} {} total latency'.format(total_latency, unit))
    if stat_props:
        executions_stats_str = '{}{} ({})'.format(prepend, beneath_stub, ', '.join(stat_props))
        out.Print(executions_stats_str)