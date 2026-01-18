import threading
from tensorboard import errors
def ExecutionDigests(self, run, begin, end):
    """Get ExecutionDigests.

        Args:
          run: The tfdbg2 run to get `ExecutionDigest`s from.
          begin: Beginning execution index.
          end: Ending execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
    runs = self.Runs()
    if run not in runs:
        return None
    execution_digests = self._reader.executions(digest=True)
    end = self._checkBeginEndIndices(begin, end, len(execution_digests))
    return {'begin': begin, 'end': end, 'num_digests': len(execution_digests), 'execution_digests': [digest.to_json() for digest in execution_digests[begin:end]]}