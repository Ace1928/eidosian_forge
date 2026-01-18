import threading
from tensorboard import errors
def GraphExecutionDigests(self, run, begin, end, trace_id=None):
    """Get `GraphExecutionTraceDigest`s.

        Args:
          run: The tfdbg2 run to get `GraphExecutionTraceDigest`s from.
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
        raise NotImplementedError('trace_id support for GraphExecutionTraceDigest is not implemented yet.')
    graph_exec_digests = self._reader.graph_execution_traces(digest=True)
    end = self._checkBeginEndIndices(begin, end, len(graph_exec_digests))
    return {'begin': begin, 'end': end, 'num_digests': len(graph_exec_digests), 'graph_execution_digests': [digest.to_json() for digest in graph_exec_digests[begin:end]]}