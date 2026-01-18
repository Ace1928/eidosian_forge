import threading
from tensorboard import errors
def GraphExecutionData(self, run, begin, end, trace_id=None):
    """Get `GraphExecutionTrace`s.

        Args:
          run: The tfdbg2 run to get `GraphExecutionTrace`s from.
          begin: Beginning graph-execution index.
          end: Ending graph-execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
    runs = self.Runs()
    if run not in runs:
        return None
    if trace_id is not None:
        raise NotImplementedError('trace_id support for GraphExecutionTraceData is not implemented yet.')
    digests = self._reader.graph_execution_traces(digest=True)
    end = self._checkBeginEndIndices(begin, end, len(digests))
    graph_executions = self._reader.graph_execution_traces(digest=False, begin=begin, end=end)
    return {'begin': begin, 'end': end, 'graph_executions': [graph_exec.to_json() for graph_exec in graph_executions]}